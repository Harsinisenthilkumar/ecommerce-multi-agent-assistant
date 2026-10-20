import os

from dotenv import load_dotenv

load_dotenv()


class Config:
	GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
	DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
	GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
	LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
	LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
	LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
