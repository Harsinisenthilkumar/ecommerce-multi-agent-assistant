# E-Commerce Multi-Agent Voice Assistant

A Python voice assistant that uses LiveKit for real-time audio, Deepgram for
speech recognition, Gemini 2.5 Flash through LangChain and LangGraph for
supervised tool routing, Google Cloud Text-to-Speech, and Silero voice activity
detection.

## Architecture

```text
Microphone / LiveKit room
          |
          v
 Deepgram STT + Silero VAD
          |
          v
 LangGraph supervisor (Gemini 2.5 Flash)
       /       |        \
 inventory   create    track
   tool       order     order
       \       |        /
          v
 LangGraph response stream
          |
          v
 Google Cloud TTS -> LiveKit audio + transcript
```

The supervisor selects from three tools: stock lookup, order creation, and
order tracking. Tool calls are dispatched by LangGraph's `ToolNode`. The
graph's in-memory checkpointer retains messages across turns that use the same
`thread_id`. The voice bot uses the LiveKit room SID as its conversation ID;
the console app uses one stable ID for its running session.

## Features

- Gemini-powered natural-language routing and follow-up questions.
- Inventory checks and order creation with positive-quantity, SKU, address,
  and stock validation.
- Order tracking by order ID.
- LiveKit audio and text input/output, Deepgram STT, Google Cloud TTS, and Silero VAD.
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
- API credentials for Google Gemini and Deepgram, plus Google Cloud
  Text-to-Speech credentials.
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
GOOGLE_API_KEY=your-google-api-key

# Optional model and voice configuration
GEMINI_MODEL=gemini-1.5-flash
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=en-US
GOOGLE_TTS_LANGUAGE=en-US
# Optional; leave unset to use the plugin's default voice
GOOGLE_TTS_VOICE=
# Optional when not using the platform's Application Default Credentials
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-cloud-service-account.json
LIVEKIT_AGENT_NAME=ecommerce-voice-assistant
```

`GEMINI_API_KEY` is also accepted in place of `GOOGLE_API_KEY`.
Google Cloud TTS uses Application Default Credentials. Configure a service
account file via `GOOGLE_APPLICATION_CREDENTIALS` or provide ADC through the
deployment platform's workload identity. The Gemini API key alone does not
authenticate Google Cloud Text-to-Speech.

## Run

Run the text-only console client:

```powershell
python main.py
```

Run the LiveKit voice worker in development mode:

```powershell
python bot.py dev
```

Run it in production mode using the LiveKit worker CLI:

```powershell
python bot.py start
```

Connect a LiveKit room client with microphone audio enabled. The assistant
transcribes the user's speech, routes the text through LangGraph, then streams
the response text to LiveKit's TTS pipeline and publishes the generated audio
and transcript.

## Tests

Run the workflow smoke test:

```powershell
python test_e2e.py
```

The test invokes Gemini and therefore requires `GOOGLE_API_KEY` or
`GEMINI_API_KEY`. It uses the in-memory demo tools and one checkpointer thread
for its sample conversation.

## Operational notes

- The LangGraph checkpointer is in-memory only. Use a durable LangGraph
  checkpointer (for example, Postgres) for restart-safe sessions.
- Keep a stable, distinct `thread_id` for each customer conversation to prevent
  conversations from sharing history.
- Avoid logging credentials or customer delivery addresses. Add appropriate
  authentication, authorization, privacy, retention, and observability controls
  before production deployment.
- The voice worker requires network access to LiveKit, Deepgram, Google, and
  Google Cloud Text-to-Speech; text-only console mode only needs Gemini.
