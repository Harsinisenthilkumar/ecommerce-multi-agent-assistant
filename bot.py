import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
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
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins import google, silero

from agents.graph import app_graph

load_dotenv()
logger = logging.getLogger("ecommerce-voice-assistant")


def _required_environment() -> dict[str, str]:
    """Validate service credentials before opening a LiveKit session."""
    google_api_key = next(
        (
            value.strip()
            for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY")
            if (value := os.getenv(name, "").strip())
        ),
        "",
    )
    credentials = {
        "LIVEKIT_URL": os.getenv("LIVEKIT_URL", "").strip(),
        "LIVEKIT_API_KEY": os.getenv("LIVEKIT_API_KEY", "").strip(),
        "LIVEKIT_API_SECRET": os.getenv("LIVEKIT_API_SECRET", "").strip(),
        "GOOGLE_API_KEY": google_api_key,
    }
    missing = [
        name
        for name, value in credentials.items()
        if not value.strip()
    ]
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


def _latest_user_text(chat_ctx: llm.ChatContext) -> tuple[str, str]:
    for item in reversed(chat_ctx.items):
        if item.type == "message" and item.role == "user":
            text = item.text_content.strip()
            if text:
                return text, f"livekit-{item.id or uuid4().hex}"
    raise ValueError("The incoming LiveKit turn does not contain user text.")


def _assistant_reply(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            text = _text_from_content(message.content).strip()
            if text:
                return text
    raise RuntimeError("LangGraph completed without producing an assistant text reply.")


async def _handle_session_text(text: str, session: AgentSession, thread_id: str) -> None:
    """Process text typed into the LiveKit console chat box."""
    try:
        result = await app_graph.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        reply = _assistant_reply(result["messages"])
        speech_handle = session.say(reply, allow_interruptions=True)
        await speech_handle.wait_for_playout()
    except Exception:
        logger.exception("Failed to process session text message.")


class LangGraphLLM(llm.LLM):
    """LiveKit LLM adapter that runs the conversation through the LangGraph."""

    def __init__(self, thread_id: str) -> None:
        super().__init__()
        self.thread_id = thread_id

    @property
    def model(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def provider(self) -> str:
        return "google"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> llm.LLMStream:
        if kwargs:
            logger.debug(
                "LangGraph adapter does not use LiveKit-specific LLM options: %s",
                ", ".join(sorted(kwargs)),
            )
        return _LangGraphStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _LangGraphStream(llm.LLMStream):
    async def _run(self) -> None:
        try:
            user_text, message_id = _latest_user_text(self._chat_ctx)
            config = {"configurable": {"thread_id": self._llm.thread_id}}
            graph_input = {
                "messages": [HumanMessage(id=message_id, content=user_text)]
            }
            response_started = False

            async for message_chunk, metadata in app_graph.astream(
                graph_input,
                config=config,
                stream_mode="messages",
            ):
                if not isinstance(message_chunk, AIMessage):
                    continue
                if getattr(message_chunk, "tool_call_chunks", None):
                    continue

                text = _text_from_content(message_chunk.content)
                if not text:
                    continue

                response_started = True
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=message_id,
                        delta=llm.ChoiceDelta(content=text),
                    )
                )

            if not response_started:
                raise RuntimeError("LangGraph completed without producing a text reply.")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("LangGraph failed while generating a voice response.")
            raise


async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession | None = None
    try:
        credentials = _required_environment()
        os.environ["GOOGLE_API_KEY"] = credentials["GOOGLE_API_KEY"]
        await ctx.connect()
        thread_id = ctx.room.sid
        tts_options = {
            "language": os.getenv("GOOGLE_TTS_LANGUAGE", "en-US").strip() or "en-US"
        }
        voice_name = os.getenv("GOOGLE_TTS_VOICE", "").strip()
        if voice_name:
            tts_options["voice_name"] = voice_name

        stt = google.STT(languages=os.getenv("DEEPGRAM_LANGUAGE", "en-US"))
        vad_instance = await asyncio.to_thread(silero.VAD.load)

        session = AgentSession(
            stt=stt,
            llm=LangGraphLLM(thread_id=thread_id),
            tts=google.TTS(**tts_options),
            vad=vad_instance,
            allow_interruptions=True,
            min_interruption_duration=0.5,
            min_interruption_words=1,
        )

        def on_voice_transcription(event: Any) -> None:
            try:
                logger.debug(
                    "Received LiveKit voice transcription event (%s).",
                    type(event).__name__,
                )
            except Exception:
                logger.exception("Failed to handle a voice transcription event.")

        def on_voice_pipeline_error(event: Any) -> None:
            try:
                error = getattr(event, "error", event)
                logger.error("LiveKit voice pipeline reported an error: %s", error)
            except Exception:
                logger.exception("Failed to handle a LiveKit voice pipeline error.")

        session.on("user_input_transcribed", on_voice_transcription)
        session.on("error", on_voice_pipeline_error)

        @session.on("message_received")
        def on_message_received(msg: llm.ChatMessage) -> None:
            if msg.role == "user" and msg.text_content:
                asyncio.create_task(
                    _handle_session_text(msg.text_content, session, thread_id)
                )

        assistant = Agent(
            instructions=(
                "You are a concise e-commerce voice assistant. Wait for the user "
                "to speak, then answer using the connected LangGraph."
            ),
            allow_interruptions=True,
        )
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=room_io.RoomInputOptions(
                audio_enabled=True,
                text_enabled=True,
                audio_sample_rate=16000,
                audio_num_channels=1,
            ),
            room_output_options=room_io.RoomOutputOptions(
                audio_enabled=True,
                transcription_enabled=True,
                audio_sample_rate=24000,
                audio_num_channels=1,
            ),
        )

        logger.info("Voice assistant started for room %s.", ctx.room.name)
    except asyncio.CancelledError:
        if session is not None:
            await session.aclose()
        raise
    except Exception:
        logger.exception("Unable to start the e-commerce voice assistant.")
        if session is not None:
            await session.aclose()
        raise


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="A_SrvLCzYYNpNu",  
            port=int(os.getenv("PORT", 8080)),
        )
    )