# E-Commerce Multi-Agent Voice Assistant

A full-stack Python voice assistant with a local browser client. LiveKit carries
audio and chat, Deepgram handles speech recognition and synthesis, Groq's Llama
3.1 8B Instant provides responses, and Silero detects voice turns.

## Architecture

```text
Microphone / LiveKit room
          |
          v
 Deepgram STT + Silero VAD
          |
          v
 LangChain ChatGroq
 (Llama 3.1 8B Instant)
          |
          v
 Deepgram TTS -> LiveKit audio + transcript
```

The browser requests short-lived LiveKit room tokens from the local FastAPI
server. API credentials remain on the server. The worker accepts microphone
audio and LiveKit chat text, then streams response audio and synchronized
transcriptions back to the room. The separate text-only console app uses the
LangGraph supervisor and its demo inventory/order tools.

## Features

- Groq-powered voice responses and LangGraph tool routing in the console app.
- Inventory checks and order creation with positive-quantity, SKU, address,
  and stock validation.
- Order tracking by order ID.
- Local browser chat and microphone support, Deepgram STT/TTS, and Silero VAD.
- Interruption support so users can speak while the assistant is responding.
- Conversation state retained for the lifetime of the process.

### Demo data and persistence

`agents/tools.py` currently implements a thread-safe, in-memory demo store. It
starts with `SKU-101` (15 units), `SKU-102` (0 units), and `SKU-103` (8 units),
as well as a sample order `ORD-88492`. New orders decrement stock and receive
generated order IDs. Data is lost when the Python process exits. Replace these
tool implementations with authenticated inventory/order service calls and a
persistent store before using the project for real commerce; this demo does
not take payment, reserve real stock, or dispatch shipments.

## Requirements

- Python 3.10 or newer.
- A LiveKit Cloud project or self-hosted LiveKit server.
- API credentials for LiveKit, Groq, and Deepgram.
- The Python dependencies in `requirements.txt`.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root. Do not commit this file or place
credentials in source code:

```dotenv
LIVEKIT_URL=wss://your-livekit-host
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret
DEEPGRAM_API_KEY=your-deepgram-api-key
GROQ_API_KEY=your-groq-api-key

# Optional voice configuration
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=en-US
```

## Run

Start the LiveKit worker in a terminal:

```powershell
python bot.py dev
```

In a second terminal, start the local web app:

```powershell
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), enter a room name, and
select **Connect**. Allow microphone access to talk; the message box sends chat
text into the same LiveKit room. The page displays your messages and the
assistant's synchronized text while playing its audio response.

For the browser to reach LiveKit, set `LIVEKIT_URL` to a URL reachable from the
browser (usually a `wss://` LiveKit Cloud URL). Keep API keys and secrets in
`.env`; the frontend receives a room-scoped, one-hour token from `/api/token`.
The worker must be running and registered with the same agent name configured
in `bot.py` (`CA_uV5hQQhZbLRF`). The room token requests dispatch of this agent
when the room is created.

Run the separate text-only console client if needed:

```powershell
python main.py
```

## Tests

Run the workflow smoke test:

```powershell
python test_e2e.py
```

The test invokes Groq and therefore requires `GROQ_API_KEY`. It uses the
in-memory demo tools and one checkpointer thread for its sample conversation.

## Operational notes

- The LangGraph checkpointer is in-memory only. Use a durable LangGraph
  checkpointer (for example, Postgres) for restart-safe sessions.
- Keep a stable, distinct `thread_id` for each customer conversation to prevent
  conversations from sharing history.
- Avoid logging credentials or customer delivery addresses. Add appropriate
  authentication, authorization, privacy, retention, and observability controls
  before production deployment.
- The browser loads the LiveKit JavaScript client from jsDelivr and requires
  network access to LiveKit, Deepgram, and Groq.
