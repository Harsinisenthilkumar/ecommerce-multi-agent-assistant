# LiveKit LangGraph E-Commerce Assistant

This project routes text and voice through one LangGraph multi-agent workflow:

- **Text input:** reliable LiveKit data packets (`chat.user`) -> supervisor ->
  specialist/tool when needed -> text reply (`assistant.chat`) in the browser.
  This path does not run TTS.
- **Voice input:** user activates the microphone -> LiveKit audio -> Silero VAD
  and Deepgram STT -> the same LangGraph workflow -> Deepgram TTS and a
  synchronized transcript.
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

Open <http://127.0.0.1:8000>, enter a room name, and connect. The microphone
stays off until you click **Mic**. Use **Send** for a text-only reply, or
activate the microphone to speak and hear a spoken reply.

The `LIVEKIT_URL` must be reachable from both the worker and the browser. The
web app creates a room-scoped token and dispatches the worker registered as
`CA_uV5hQQhZbLRF`.

## Run the workflow smoke test

```powershell
.\.venv\Scripts\Activate.ps1
python test_e2e.py
```

The smoke test makes live Groq requests. It uses the in-memory demo tools and
does not require a LiveKit room or Deepgram request.
