import asyncio
import logging
import os
from typing import Any

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    room_io,
)
from livekit.plugins import deepgram, openai, silero

from config import Config

logger = logging.getLogger("ecommerce-voice-assistant")


def _required_environment() -> dict[str, str]:
    """Validate service credentials before opening a LiveKit session."""
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


async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession | None = None
    try:
        credentials = _required_environment()
        await ctx.connect()

        stt = deepgram.STT(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language=os.getenv("DEEPGRAM_LANGUAGE", "en-US"),
            api_key=credentials["DEEPGRAM_API_KEY"],
        )
        tts = deepgram.TTS(
            model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en"),
            api_key=credentials["DEEPGRAM_API_KEY"],
        )
        vad_instance = await asyncio.to_thread(silero.VAD.load)
        groq_llm = openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=credentials["GROQ_API_KEY"],
            model="llama-3.1-8b-instant",
        )

        session = AgentSession(
            stt=stt,
            llm=groq_llm,
            tts=tts,
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

        assistant = Agent(
            instructions=(
                "You are a concise e-commerce assistant. You do not have access "
                "to inventory, ordering, or order-tracking tools. Do not claim "
                "to check stock, place orders, or retrieve order status."
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