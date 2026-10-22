import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import os
from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli, llm, VoicePipelineAgent
from livekit.plugins import deepgram, groq, silero

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

def entrypoint(ctx: JobContext):
    logger.info(f"*** Agent connected to room: {ctx.room.name} ***")
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="You are a helpful e-commerce customer support assistant. Help users track orders and check stock. Reply concisely.",
    )

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=groq.LLM(model=os.getenv("GROQ_MODEL", "llama3-8b-8192")),
        tts=deepgram.TTS(),
        chat_ctx=initial_ctx,
    )

    # Listen to text/data messages sent from the UI chat box
    @ctx.room.on("data_received")
    def on_data_received(data: bytes, participant, **kwargs):
        try:
            text = data.decode("utf-8")
            logger.info(f">>> Received text from UI: {text}")
            agent.chat_ctx.append(role="user", text=text)
            asyncio.create_task(agent.generate_reply())
        except Exception as e:
            logger.error(f"Error handling data message: {e}")

    agent.start(ctx.room)
    logger.info("*** VoicePipelineAgent started successfully! ***")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))