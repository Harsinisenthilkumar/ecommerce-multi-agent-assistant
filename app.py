import logging
import re
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from livekit import api
from pydantic import BaseModel, Field

from config import Config

load_dotenv()

logger = logging.getLogger("ecommerce-voice-frontend")
app = FastAPI(title="E-Commerce Voice Assistant")
INDEX_FILE = Path(__file__).parent / "frontend" / "index.html"
ROOM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
AGENT_NAME = Config.AGENT_NAME


class TokenRequest(BaseModel):
    room: str = Field(min_length=1, max_length=128)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.post("/api/token")
async def create_token(request: TokenRequest) -> dict[str, str]:
    conversation_label = request.room.strip()
    if not ROOM_NAME_PATTERN.fullmatch(conversation_label):
        raise HTTPException(
            status_code=422,
            detail=(
                "Conversation labels may contain only letters, numbers, "
                "underscores, and hyphens."
            ),
        )
    conversation_id = uuid4().hex
    room = f"{conversation_label[:80]}-{conversation_id}"

    livekit_url = Config.LIVEKIT_URL.strip()
    api_key = Config.LIVEKIT_API_KEY.strip()
    api_secret = Config.LIVEKIT_API_SECRET.strip()
    missing = [
        name
        for name, value in (
            ("LIVEKIT_URL", livekit_url),
            ("LIVEKIT_API_KEY", api_key),
            ("LIVEKIT_API_SECRET", api_secret),
        )
        if not value
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail="LiveKit is not configured. Set " + ", ".join(missing) + " in .env.",
        )

    identity = f"web-{uuid4().hex[:16]}"
    try:
        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name("Web visitor")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .with_room_config(
                api.RoomConfiguration(
                    agents=[
                        api.RoomAgentDispatch(
                            agent_name=AGENT_NAME,
                            metadata=conversation_id,
                        )
                    ]
                )
            )
            .with_ttl(timedelta(hours=1))
            .to_jwt()
        )
    except Exception as exc:
        logger.exception("Failed to generate a LiveKit room token.")
        raise HTTPException(
            status_code=500,
            detail="Could not create a LiveKit token.",
        ) from exc

    return {
        "url": livekit_url,
        "token": token,
        "room": room,
        "identity": identity,
        "conversation_id": conversation_id,
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
