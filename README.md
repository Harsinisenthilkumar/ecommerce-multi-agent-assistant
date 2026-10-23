# LiveKit LangGraph E-Commerce Assistant

This project routes text and voice through one LangGraph multi-agent workflow:

- **Text input:** reliable LiveKit data packets (`chat.user`) -> supervisor ->
  specialist/tool when needed -> text reply (`assistant.chat`) in the browser.
  This path does not run TTS.
- **Voice input:** user activates the microphone -> LiveKit audio -> Silero VAD
  and Deepgram STT -> the same LangGraph workflow -> Deepgram TTS and a
  synchronized transcript.
- **Browser voice fallback:** **Web voice** uses browser speech recognition and
  speech synthesis, sending recognized speech over the same reliable text-chat
  path. It is separate from **Mic** and does not change typed chat. Speech
  recognition availability and processing depend on browser support and
  browser privacy settings; a network connection may be required.
- **Conversation RAG:** each browser connection gets a unique conversation ID
  and a private LiveKit room. Successful text and voice turns are embedded
  locally and stored in `data/conversation_memory.sqlite3`; relevant turns are
  retrieved only within that conversation. Reconnecting starts a new
  conversation and cannot retrieve the previous connection's memories. The
  local BGE model is downloaded on first use to
  `~/.cache/ecommerce-multi-agent/fastembed` (override with
  `CONVERSATION_EMBEDDING_CACHE`). If memory storage or embedding is unavailable,
  the assistant logs the issue and continues without retrieved history.
- **Agents:** the LangGraph supervisor routes inventory, order creation, and
  order tracking requests to separate specialist agents. General questions are
  answered by the supervisor.

The demo commerce tools use an in-memory store. They do not take payment,
reserve real stock, or dispatch shipments.

## Requirements

- Python 3.10 or newer.
- A LiveKit Cloud project or self-hosted server.
- LiveKit, Groq, and Deepgram credentials.

## Configure

From PowerShell in the repository directory, install dependencies in a virtual
environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Add these values to `.env` in the repository root. Keep the real credentials
private; do not commit `.env`.

```dotenv
LIVEKIT_URL=wss://your-livekit-host
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret
GROQ_API_KEY=your-groq-api-key
DEEPGRAM_API_KEY=your-deepgram-api-key

# Optional voice settings
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=en-US
DEEPGRAM_TTS_MODEL=aura-asteria-en
```

The Groq model is configured in `config.py` as `openai/gpt-oss-20b`. The
previous `llama3-8b-8192` model is retired; Groq lists its shutdown date as
August 30, 2025.

## Run locally

Open **two PowerShell terminals**, each in the repository directory.

Terminal 1 — LiveKit worker:

```powershell
.\.venv\Scripts\Activate.ps1
python bot.py dev
```

Terminal 2 — web interface and token service:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Open <http://127.0.0.1:8000>, enter a conversation label, and connect. Each
connection creates a private LiveKit room and a fresh conversation ID; the
label is only used to identify the session. The microphone stays off until you
click **Mic**. Use **Send** for a text-only reply, **Mic** for LiveKit voice, or
**Web voice** to use browser speech recognition and speech playback. Browser
voice requires a supported browser and microphone permission.

The `LIVEKIT_URL` must be reachable from both the worker and the browser. The
web app creates a room-scoped token and dispatches the worker registered as
`CA_uV5hQQhZbLRF`.

Conversation memories contain user messages and assistant replies. They are
stored locally, scoped by a unique ID generated for each browser connection,
and retained until manually deleted. The UI shows the first eight characters
of the ID; the worker logs the full ID when a session starts. To delete one
conversation, run this in another PowerShell window from the repository:

```powershell
.\.venv\Scripts\python.exe -m agents.conversation_memory --clear-conversation <conversation-id>
```

The `--clear-room` option remains available to remove legacy room-scoped records
from the earlier memory format.

Set `CONVERSATION_MEMORY_DB` to use a different local SQLite file. The database
is ignored by Git and should not be shared or committed.

## Run the workflow smoke test

```powershell
.\.venv\Scripts\Activate.ps1
python test_e2e.py
```

The smoke test makes live Groq requests. It uses the in-memory demo tools and
does not require a LiveKit room or Deepgram request.
