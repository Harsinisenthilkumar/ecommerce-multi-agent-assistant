import os

from dotenv import load_dotenv

load_dotenv()


class Config:
	DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
	GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
	LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
	LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
	LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
