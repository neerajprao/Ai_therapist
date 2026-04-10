# SoulSync — AI Therapy Voice Assistant

SoulSync is an interactive AI therapist prototype built with a Flask web UI, local speech-to-text, an LLM-based therapy response engine, emotion-aware voice synthesis, and a hybrid emotion analysis pipeline.

## Project Overview

- **Frontend**: A Tailwind-based single-page web interface at `templates/index.html` with JavaScript audio recording and real-time streaming.
- **Backend**: A Flask server in `app.py` that accepts recorded audio, transcribes it, generates a therapeutic reply, and returns voice audio.
- **Speech recognition**: `whisper` is used locally inside `brain.py` for transcription (base model for efficiency).
- **LLM model**: `ollama` calls a local `llama3` model to generate empathetic therapist responses as Luna, a soft-spoken therapist.
- **Voice rendering**: Uses the Inworld TTS API to produce a calm, emotional voice from generated text, with speed and temperature adjustments based on detected emotion.
- **Emotion analysis**: Two-tier system: text-based emotion detection in `brain.py` for TTS prompts, and a trained voice emotion classifier in `therapist_bot.py` using `Emotion2Vec` embeddings and a bridge neural network.
- **Session management**: Each user session gets a unique ID, with conversation history saved as JSON files in `data/history/`.

## Repository Structure

- `app.py` — Flask application with routes for UI, audio processing, and TTS. Handles session creation, audio saving, streaming responses, and Inworld API calls.
- `brain.py` — Core logic: Whisper transcription, text-based emotion detection, LLM streaming chat with Ollama, history management per session.
- `therapist_bot.py` — Voice emotion classification wrapper using Emotion2Vec encoder and trained bridge network for 5 emotion categories.
- `train_emotion.py` — Loads `models/emotion2vec` via FunASR AutoModel, extracts emotion labels and 1024D embeddings from audio.
- `train_bridge.py` — Defines and trains the `TherapistBridge` neural network (1024→256→5) to map embeddings to emotion classes.
- `generate_synthetic_data.py` — Builds synthetic training data from anchor audio files by adding Gaussian noise to create larger datasets.
- `prep_data.py` — Creates a skeleton `data/metadata.csv` for labeling audio files (currently unused in pipeline).
- `transcribe.py` — Standalone Whisper transcription test harness.
- `requirements.txt` — Python dependencies including torch, funasr, librosa, flask, requests, python-dotenv, ollama, whisper.
- `how_to_open.txt` — Quick start commands.
- `templates/index.html` — Frontend UI with Tailwind CSS, audio recording via MediaRecorder API, streaming text display, and dynamic emotion-based theming.
- `data/` — Raw audio files, saved recordings, synthetic data (X_train.npy, y_train.npy), test embeddings, session history JSONs.
- `models/` — Pretrained Emotion2Vec model (config.yaml, model.pt, tokens.txt), saved bridge checkpoint (bridge_v1.pth).
- `static/` — Generated TTS audio files (MP3s).

## How It Works

1. User visits the web interface; a new session is created with a unique ID and initial history JSON.
2. User clicks record; browser captures microphone input via MediaRecorder API.
3. Frontend sends raw audio blob to `/process_audio_stream`.
4. `app.py` saves audio to `data/raw_audio/{session_id}_input.wav`, transcribes via `brain.py`, detects text emotion.
5. Stream starts with metadata: `METADATA|{user_text}|{detected_emotion}|`, then LLM-generated text chunks.
6. Frontend displays user text, emotion, and streams AI response.
7. After streaming, frontend calls `/get_audio` with full text and emotion.
8. `app.py` adjusts TTS parameters (speed, temperature, prompt) based on emotion, calls Inworld API, saves MP3 to `static/audio/`, returns URL.
9. Frontend plays the audio and fades in the text.

## Key Features

- **Local STT** using Whisper base model: no cloud required, runs on CPU/MPS.
- **LLM empathy persona**: Luna responds briefly (1-2 sentences), uses pauses, removes actions/stage directions.
- **Real-time streaming**: Partial text streamed as model generates; history saved per session.
- **Emotion-aware TTS**: Voice prompt modified for Sad (whispering, sighing), Anxious (soothing, breathing), Neutral (calm); speed/temperature tuned.
- **Hybrid emotion analysis**: Text keywords for TTS, trained model for voice classification (Anxious, Sad, Angry, Neutral, Happy).
- **Session persistence**: Conversation history maintained in JSON files, loaded per session.

## Installation

### Prerequisites

- Python 3.8+
- Ollama installed and `llama3` model pulled (`ollama pull llama3`)
- Inworld TTS API key (sign up at inworld.ai)
- Audio files for training/testing (e.g., `data/raw_audio/test.wav`, `data/raw_audio/sad.wav`)

### 1. Clone repository

```bash
git clone <repo-url>
cd Ai_therapist
```

### 2. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Note: This installs torch (with MPS support for Apple Silicon), funasr for Emotion2Vec, librosa for audio, flask for web, etc.

### 4. Create environment file

Create a `.env` file in the repository root with your Inworld authorization key:

```text
INWORLD_KEY=your_base64_encoded_inworld_basic_auth_key
```

- Obtain from Inworld dashboard; it's Base64 encoded `username:password`.
- Used by `app.py` for TTS API calls.

### 5. Ensure local models are available

- `whisper` loads `base` model automatically.
- `ollama` must have `llama3` available.
- `models/emotion2vec/` must contain the full model files (downloaded separately, not included in repo).

## Running the App

```bash
source venv/bin/activate
python app.py
```

- Starts Flask on port 5000.
- Open `http://127.0.0.1:5000` in browser.
- Allow microphone access when prompted.
- Click the record button to start therapy session.

### Available Endpoints

- `GET /` — Serves the main UI, initializes new session with history JSON.
- `POST /process_audio_stream` — Accepts audio file, returns streaming text with initial metadata.
- `POST /get_audio` — Accepts text and emotion, returns TTS audio URL.

## Training and Data Pipeline

### Prepare metadata (optional)

```bash
python prep_data.py
```

Creates `data/metadata.csv` with columns `file_path` and `label` (for future expansion).

### Generate synthetic training examples

```bash
python generate_synthetic_data.py
```

- Loads Emotion2Vec processor.
- Extracts embeddings from anchor files (`data/raw_audio/test.wav` → Happy, `data/raw_audio/sad.wav` → Sad).
- Generates 100 noisy variations per anchor (Gaussian noise σ=0.02).
- Saves `data/X_train.npy` (embeddings) and `data/y_train.npy` (labels).

### Train the bridge network

```bash
python train_bridge.py
```

- Loads synthetic data.
- Trains `TherapistBridge` (1024→256→5) with Adam optimizer, NLLLoss, 50 epochs.
- Saves weights to `models/checkpoints/bridge_v1.pth`.

### Test Emotion Embeddings

```bash
python train_emotion.py
```

- Loads Emotion2Vec model on MPS/CPU.
- Processes `data/raw_audio/test.wav`, prints top emotion and embedding shape.
- Saves embedding to `data/test_embedding.npy`.

### Test Voice Emotion Classification

```bash
python therapist_bot.py
```

- Loads encoder and bridge.
- Analyzes `data/raw_audio/sad.wav`, prints predicted emotion from 5 classes.

### Test Transcription

```bash
python transcribe.py
```

- Loads Whisper base model.
- Transcribes `data/raw_audio/sad.wav`, prints text.

## Audio Emotion Inference

`therapist_bot.py` provides voice-based emotion prediction:

- Loads Emotion2Vec encoder and trained bridge.
- Predicts one of: Anxious/Stressed, Sad/Depressed, Angry/Frustrated, Neutral/Calm, Happy/Stable.

Text-based emotion in `brain.py` uses keywords for TTS (Sad, Anxious, Neutral).

## Frontend Details

- Built with Tailwind CSS for glassmorphism design.
- Uses MediaRecorder for audio capture.
- Streams response text, updates emotion card with color themes.
- Plays TTS audio after generation.
- Session-based, refreshes create new history.

## Required Files and Directories

- `data/raw_audio/` — Anchor audio files for training (e.g., test.wav, sad.wav).
- `data/X_train.npy` / `data/y_train.npy` — Synthetic training data.
- `data/test_embedding.npy` — Sample embedding.
- `data/history/` — Per-session JSON conversation logs.
- `models/emotion2vec/` — Emotion2Vec model directory.
- `models/checkpoints/bridge_v1.pth` — Trained bridge weights.
- `static/audio/` — Generated TTS MP3s.
- `.env` — Inworld API key.

## Dependencies

Listed in `requirements.txt`:

- `torch`, `torchvision`, `torchaudio` — PyTorch ecosystem.
- `funasr`, `modelscope` — For Emotion2Vec.
- `librosa`, `numpy`, `pandas`, `soundfile` — Audio processing.
- `flask`, `requests`, `python-dotenv` — Web and config.
- `ollama` — LLM interface.
- `whisper` — Speech recognition.

## Known Caveats

- Requires Inworld API key for TTS.
- Emotion classifier is prototype with limited synthetic data.
- Text emotion detection is simple keyword-based.
- Frontend expects `METADATA|` prefix in stream for emotion display.
- Whisper and Ollama run locally; ensure sufficient RAM/CPU.
- Tested on macOS with MPS; adjust device in code for other platforms.

## Troubleshooting

- **Whisper/Ollama not found**: Ensure models are downloaded/installed.
- **Inworld TTS fails**: Check `.env` key format and API limits.
- **Training errors**: Verify `models/emotion2vec/` exists and audio files are present.
- **Audio not recording**: Check browser microphone permissions.
- **Session history not saving**: Ensure `data/history/` is writable.

## Recommended Next Steps

1. Collect real labeled audio for better emotion training.
2. Expand synthetic data generation to all 5 emotion classes.
3. Integrate voice emotion analysis into main app flow.
4. Add user authentication and session management.
5. Improve text emotion detection with NLP models.
6. Add error handling and logging.
7. Deploy with Docker for easier setup.

## License

This project does not include a license file by default. Add `LICENSE` if you want to specify reuse terms.
