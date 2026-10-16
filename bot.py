import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
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
from livekit.plugins import deepgram, openai, silero

from agents.graph import app_graph

load_dotenv()
logger = logging.getLogger("ecommerce-voice-assistant")


def _required_environment() -> dict[str, str]:
    """Validate service credentials before opening a LiveKit session."""
    credentials = {
        "LIVEKIT_URL": os.getenv("LIVEKIT_URL", ""),
        "LIVEKIT_API_KEY": os.getenv("LIVEKIT_API_KEY", ""),
        "LIVEKIT_API_SECRET": os.getenv("LIVEKIT_API_SECRET", ""),
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY", ""),
    }
    missing = [name for name, value in credentials.items() if not value.strip()]
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
                if metadata.get("langgraph_node") != "supervisor":
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
        await ctx.connect()
        session = AgentSession(
            stt=deepgram.STT(
                model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
                language=os.getenv("DEEPGRAM_LANGUAGE", "en-US"),
                api_key=credentials["DEEPGRAM_API_KEY"],
            ),
            llm=LangGraphLLM(thread_id=ctx.room.sid),
            tts=openai.TTS(
                model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
                voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
                api_key=credentials["OPENAI_API_KEY"],
            ),
            vad=silero.VAD.load(),
            allow_interruptions=True,
            min_interruption_duration=0.5,
            min_interruption_words=1,
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
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", "ecommerce-voice-assistant"),
        )
    )
