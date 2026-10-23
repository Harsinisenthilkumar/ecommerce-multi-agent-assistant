import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    room_io,
)
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.plugins import deepgram, silero

from agents.conversation_memory import (
    ConversationMemoryError,
    ConversationMemoryStore,
)
from agents.graph import app_graph
from config import Config

load_dotenv()

logger = logging.getLogger("ecommerce-livekit-agent")
conversation_memory = ConversationMemoryStore()
AGENT_NAME = Config.AGENT_NAME
USER_CHAT_TOPIC = "chat.user"
ASSISTANT_CHAT_TOPIC = "assistant.chat"
MAX_CHAT_LENGTH = 2000


def _required_environment() -> dict[str, str]:
    credentials = {
        "LIVEKIT_URL": Config.LIVEKIT_URL.strip(),
        "LIVEKIT_API_KEY": Config.LIVEKIT_API_KEY.strip(),
        "LIVEKIT_API_SECRET": Config.LIVEKIT_API_SECRET.strip(),
        "DEEPGRAM_API_KEY": Config.DEEPGRAM_API_KEY.strip(),
        "GROQ_API_KEY": Config.GROQ_API_KEY.strip(),
    }
    missing = [name for name, value in credentials.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return credentials


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, str)
            or (isinstance(part, dict) and isinstance(part.get("text"), str))
        )
    return ""


def _latest_user_text(chat_ctx: llm.ChatContext) -> str:
    provider_messages, _ = chat_ctx.to_provider_format(
        "openai",
        inject_dummy_user_message=False,
    )
    for message in reversed(provider_messages):
        if message.get("role") == "user":
            return _text_from_content(message.get("content")).strip()
    return ""


async def _run_graph_turn(
    text: str,
    conversation_id: str,
    turn_lock: asyncio.Lock,
) -> str:
    if not text.strip():
        raise ValueError("Cannot run a LangGraph turn without user text.")

    async with turn_lock:
        try:
            memories = await asyncio.to_thread(
                conversation_memory.retrieve,
                conversation_id,
                text.strip(),
            )
        except ConversationMemoryError as exc:
            logger.exception(
                "Conversation memory retrieval failed for conversation %s; "
                "continuing without retrieved history: %s",
                conversation_id,
                exc,
            )
            memories = []

        result = await app_graph.ainvoke(
            {
                "messages": [HumanMessage(content=text.strip())],
                "retrieved_memories": memories,
            },
            config={"configurable": {"thread_id": conversation_id}},
        )

    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", None) == "ai":
            answer = _text_from_content(getattr(message, "content", "")).strip()
            if answer:
                try:
                    await asyncio.to_thread(
                        conversation_memory.save_turn,
                        conversation_id,
                        text.strip(),
                        answer,
                    )
                except ConversationMemoryError as exc:
                    logger.exception(
                        "Conversation memory save failed for conversation %s; "
                        "the response was still generated: %s",
                        conversation_id,
                        exc,
                    )
                return answer
    raise RuntimeError("LangGraph completed without an assistant text response.")


def _decode_text_request(packet: rtc.DataPacket) -> tuple[str, str] | None:
    if packet.topic != USER_CHAT_TOPIC or packet.participant is None:
        return None

    try:
        request = json.loads(packet.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Chat packet must contain valid UTF-8 JSON.") from exc

    if not isinstance(request, dict) or request.get("type") != "chat":
        raise ValueError("Chat packet has an invalid message type.")
    text = request.get("text")
    request_id = request.get("request_id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Chat text must be a non-empty string.")
    if len(text) > MAX_CHAT_LENGTH:
        raise ValueError(f"Chat text must be at most {MAX_CHAT_LENGTH} characters.")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 64:
        raise ValueError("Chat request ID is missing or invalid.")
    return text.strip(), request_id


async def _publish_text_response(
    room: rtc.Room,
    request_id: str,
    text: str,
) -> None:
    payload = json.dumps(
        {
            "type": "assistant_response",
            "request_id": request_id,
            "text": text,
        }
    )
    await room.local_participant.publish_data(
        payload,
        reliable=True,
        topic=ASSISTANT_CHAT_TOPIC,
    )


class LangGraphLLM(llm.LLM):
    """Expose the shared LangGraph workflow as a LiveKit speech LLM."""

    def __init__(
        self,
        conversation_id: str,
        turn_lock: asyncio.Lock,
    ) -> None:
        super().__init__()
        self._conversation_id = conversation_id
        self._turn_lock = turn_lock

    @property
    def model(self) -> str:
        return Config.GROQ_MODEL

    @property
    def provider(self) -> str:
        return "langgraph"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        if tools:
            logger.warning("LiveKit tools are ignored; commerce tools run in LangGraph.")
        if parallel_tool_calls is not NOT_GIVEN or tool_choice is not NOT_GIVEN:
            logger.debug("Ignoring LiveKit tool-call options; LangGraph owns tool routing.")
        if extra_kwargs is not NOT_GIVEN:
            logger.debug("Ignoring unsupported LiveKit LLM options.")
        return _LangGraphLLMStream(
            self,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
        )


class _LangGraphLLMStream(llm.LLMStream):
    def __init__(
        self,
        graph_llm: LangGraphLLM,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            graph_llm,
            chat_ctx=chat_ctx,
            tools=[],
            conn_options=conn_options,
        )
        self._conversation_id = graph_llm._conversation_id
        self._turn_lock = graph_llm._turn_lock

    async def _run(self) -> None:
        try:
            user_text = _latest_user_text(self._chat_ctx)
            if not user_text:
                raise RuntimeError("No transcribed voice input reached LangGraph.")

            logger.info("Sending voice transcript to the LangGraph workflow.")
            answer = await _run_graph_turn(
                user_text,
                self._conversation_id,
                self._turn_lock,
            )
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=f"langgraph-{id(self)}",
                    delta=llm.ChoiceDelta(content=answer),
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("LangGraph voice response generation failed.")
            raise


async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession | None = None
    text_tasks: set[asyncio.Task[None]] = set()
    try:
        credentials = _required_environment()
        conversation_id = ctx.job.metadata.strip()
        if not conversation_id:
            conversation_id = f"livekit-{uuid4().hex}"
            logger.warning(
                "Agent dispatch did not include a conversation ID; "
                "using a session-only ID %s.",
                conversation_id,
            )
        turn_lock = asyncio.Lock()

        try:
            vad = await asyncio.to_thread(
                silero.VAD.load,
                min_speech_duration=0.05,
                min_silence_duration=0.5,
                prefix_padding_duration=0.3,
                max_buffered_speech=30.0,
                activation_threshold=0.5,
                sample_rate=16000,
            )
        except Exception as exc:
            logger.exception("Failed to initialize Silero VAD.")
            raise RuntimeError("Unable to initialize Silero VAD.") from exc

        stt = deepgram.STT(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language=os.getenv("DEEPGRAM_LANGUAGE", "en-US"),
            api_key=credentials["DEEPGRAM_API_KEY"],
        )
        tts = deepgram.TTS(
            model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en"),
            api_key=credentials["DEEPGRAM_API_KEY"],
        )

        async def process_text_packet(packet: rtc.DataPacket) -> None:
            request_id = "unknown"
            try:
                text_request = _decode_text_request(packet)
                if text_request is None:
                    return
                user_text, request_id = text_request
                logger.info(
                    "Received text chat request %s from participant %s.",
                    request_id,
                    packet.participant.identity,
                )
                answer = await _run_graph_turn(
                    user_text,
                    conversation_id,
                    turn_lock,
                )
                await _publish_text_response(ctx.room, request_id, answer)
            except asyncio.CancelledError:
                raise
            except ValueError:
                logger.warning("Ignoring invalid text data packet.")
            except Exception:
                logger.exception("LangGraph text response generation failed.")
                try:
                    await _publish_text_response(
                        ctx.room,
                        request_id,
                        "Sorry, I couldn't complete that text reply. Please try again.",
                    )
                except Exception:
                    logger.exception("Failed to publish the data-channel error response.")

        def on_data_received(packet: rtc.DataPacket) -> None:
            if packet.topic != USER_CHAT_TOPIC or packet.participant is None:
                return
            task = asyncio.create_task(process_text_packet(packet))
            text_tasks.add(task)
            task.add_done_callback(text_tasks.discard)

        session = AgentSession(
            stt=stt,
            llm=LangGraphLLM(conversation_id, turn_lock),
            tts=tts,
            vad=vad,
            allow_interruptions=True,
            min_interruption_duration=0.5,
            min_interruption_words=1,
            min_endpointing_delay=0.5,
            max_endpointing_delay=2.0,
        )

        def on_session_error(event: Any) -> None:
            logger.error(
                "LiveKit audio pipeline failed (STT, LangGraph, or TTS): %s",
                getattr(event, "error", event),
            )

        session.on("error", on_session_error)
        assistant = Agent(
            instructions="Handle voice turns through the shared e-commerce LangGraph.",
            allow_interruptions=True,
        )

        await ctx.connect()
        ctx.room.on("data_received", on_data_received)
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_options=room_io.RoomOptions(
                text_input=False,
                audio_input=room_io.AudioInputOptions(
                    sample_rate=16000,
                    num_channels=1,
                    frame_size_ms=50,
                ),
                audio_output=room_io.AudioOutputOptions(
                    sample_rate=24000,
                    num_channels=1,
                ),
                text_output=room_io.TextOutputOptions(sync_transcription=True),
            ),
        )
        logger.info(
            "Multi-agent assistant started in room %s with conversation %s.",
            ctx.room.name,
            conversation_id,
        )
    except asyncio.CancelledError:
        for task in text_tasks:
            task.cancel()
        if text_tasks:
            await asyncio.gather(*text_tasks, return_exceptions=True)
        if session is not None:
            await session.aclose()
        raise
    except Exception:
        logger.exception("Unable to start the LiveKit LangGraph assistant.")
        for task in text_tasks:
            task.cancel()
        if text_tasks:
            await asyncio.gather(*text_tasks, return_exceptions=True)
        if session is not None:
            await session.aclose()
        raise


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME,
            port=int(os.getenv("PORT", 8080)),
        )
    )
