import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, convert_to_messages
from langchain_groq import ChatGroq
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

from config import Config

load_dotenv()

logger = logging.getLogger("ecommerce-voice-assistant")
GROQ_MODEL = "llama-3.1-8b-instant"


def _required_environment() -> dict[str, str]:
    credentials = {
        "LIVEKIT_URL": Config.LIVEKIT_URL.strip(),
        "LIVEKIT_API_KEY": Config.LIVEKIT_API_KEY.strip(),
        "LIVEKIT_API_SECRET": Config.LIVEKIT_API_SECRET.strip(),
        "DEEPGRAM_API_KEY": Config.DEEPGRAM_API_KEY.strip(),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "").strip(),
    }
    missing = [name for name, value in credentials.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return credentials


def _get_groq_chat_model() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Groq credentials are missing. Set GROQ_API_KEY.")
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=api_key,
        max_retries=5,
    )


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


class GroqLLM(llm.LLM):
    """Adapt LangChain ChatGroq streaming to LiveKit's LLM interface."""

    def __init__(self, chat_model: ChatGroq) -> None:
        super().__init__()
        self._chat_model = chat_model

    @property
    def model(self) -> str:
        return GROQ_MODEL

    @property
    def provider(self) -> str:
        return "groq"

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
            raise ValueError("The Groq voice assistant does not support tools.")
        if (
            parallel_tool_calls is not NOT_GIVEN
            or tool_choice is not NOT_GIVEN
            or extra_kwargs is not NOT_GIVEN
        ):
            logger.debug("Ignoring tool options because the Groq adapter has no tools.")
        return _GroqLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=[],
            conn_options=conn_options,
        )


class _GroqLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        try:
            provider_messages, _ = self._chat_ctx.to_provider_format(
                "openai",
                inject_dummy_user_message=False,
            )
            messages = convert_to_messages(provider_messages)
            response_started = False
            async for chunk in self._llm._chat_model.astream(messages):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = _text_from_content(chunk.content)
                if not text:
                    continue
                response_started = True
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=f"groq-{id(self)}",
                        delta=llm.ChoiceDelta(content=text),
                    )
                )
            if not response_started:
                raise RuntimeError("Groq completed without producing a text response.")
        except Exception:
            logger.exception("Groq LLM generation failed.")
            raise


async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession | None = None
    try:
        credentials = _required_environment()

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
        groq_llm = GroqLLM(_get_groq_chat_model())
        groq_llm.on(
            "error",
            lambda event: logger.error("Groq LLM request failed: %s", event.error),
        )

        session = AgentSession(
            stt=stt,
            llm=groq_llm,
            tts=tts,
            vad=vad,
            allow_interruptions=True,
            min_interruption_duration=0.5,
            min_interruption_words=1,
            turn_handling={
                "endpointing": {
                    "mode": "fixed",
                    "min_delay": 0.5,
                    "max_delay": 2.0,
                },
            },
        )

        def on_voice_pipeline_error(event: Any) -> None:
            logger.error(
                "LiveKit voice pipeline generation failed (LLM/STT/TTS): %s",
                getattr(event, "error", event),
            )

        session.on("error", on_voice_pipeline_error)

        assistant = Agent(
            instructions=(
                "You are a concise e-commerce assistant. You do not have access "
                "to inventory, ordering, or order-tracking tools. Do not claim "
                "to check stock, place orders, or retrieve order status."
            ),
            allow_interruptions=True,
        )

        await ctx.connect()
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_options=room_io.RoomOptions(
                text_input=room_io.TextInputOptions(),
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
        logger.info("Voice and text agent started in room %s.", ctx.room.name)
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
            agent_name="CA_uV5hQQhZbLRF",
            port=int(os.getenv("PORT", 8080)),
        )
    )