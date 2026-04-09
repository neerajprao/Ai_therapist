# SoulSync — AI Therapy Voice Assistant

SoulSync is an interactive AI therapist prototype built with a Flask web UI, local speech-to-text, an LLM-based therapy response engine, emotion-aware voice synthesis, and a hybrid emotion analysis pipeline.

## Project Overview

- **Frontend**: A Tailwind-based single-page web interface at `templates/index.html`.
- **Backend**: A Flask server in `app.py` that accepts recorded audio, transcribes it, generates a therapeutic reply, and returns voice audio.
- **Speech recognition**: `whisper` is used locally inside `brain.py` for transcription.
- **LLM model**: `ollama` calls a local `llama3` model to generate empathetic therapist responses.
- **Voice rendering**: Uses the Inworld TTS API to produce a calm, emotional voice from generated text.
- **Emotion analysis**: `train_emotion.py` and `train_bridge.py` create a two-phase voice emotion classifier via `Emotion2Vec` and a lightweight bridge neural network.

## Repository Structure

- `app.py` — Flask application, REST endpoints, audio recording flow, Inworld TTS integration.
- `brain.py` — Core speech-to-text, text cleaning, LLM chat prompt logic, streaming response generation.
- `therapist_bot.py` — Emotion classification wrapper using embeddings + bridge network.
- `train_emotion.py` — Loads `models/emotion2vec`, extracts embeddings and labels from audio.
- `train_bridge.py` — Defines and trains the small bridge network mapping embeddings to 5 emotion categories.
- `generate_synthetic_data.py` — Builds synthetic training data from a few audio anchors.
- `prep_data.py` — Helper that creates a metadata CSV placeholder for audio labels.
- `transcribe.py` — Simple Whisper test harness for local transcription.
- `requirements.txt` — Package dependencies.
- `templates/index.html` — Frontend UI and JavaScript audio capture logic.
- `data/` — Raw audio files, saved recordings, synthetic data, and test metadata.
- `models/` — Pretrained Emotion2Vec model and saved bridge checkpoint.

## How It Works

1. User records audio through the browser.
2. The frontend sends the raw audio to `/process_audio_stream`.
3. `app.py` saves the file under `data/raw_audio/` and passes it to `TherapistBrain.generate_streaming_response()`.
4. `brain.py` transcribes the audio locally with Whisper.
5. The transcribed text is appended to chat history and sent to `ollama.chat(...)`.
6. The stream is cleaned to remove stage directions and asterisks.
7. The Flask endpoint returns text chunks to the browser.
8. The browser then calls `/get_audio` with the final assistant text and emotion hint.
9. `/get_audio` calls the Inworld TTS service and returns an MP3 URL.

## Key Features

- **Local STT** using Whisper: no cloud speech transcription required.
- **LLM empathy persona**: Luna is configured as a soft, concise therapist.
- **Real-time streaming**: partial text is streamed back as the model generates.
- **Emotion-aware TTS**: the voice prompt changes depending on detected mood.
- **Voice emotion analysis**: separate pipeline for analyzing emotional tone from voice embeddings.

## Installation

### 1. Clone repository

```bash
git clone <repo-url>
cd Ai_therapist
```

### 2. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create environment file

Create a `.env` file in the repository root with your Inworld authorization key:

```text
INWORLD_KEY=<your_inworld_basic_auth_key>
```

- `INWORLD_KEY` is used by `app.py` to call the Inworld TTS endpoint.

### 5. Ensure local models are available

- `whisper` will load the `base` speech-to-text model.
- `ollama` expects a local `llama3` model pulled via `ollama pull llama3`.
- `models/emotion2vec` must exist in the repository for `train_emotion.py`.

## Running the App

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

### Available Endpoints

- `GET /` — serves the web interface.
- `POST /process_audio_stream` — accepts recorded audio, transcribes it, and streams text.
- `POST /get_audio` — accepts assistant text and emotion metadata, then returns TTS audio.

## Training and Data Pipeline

### Prepare metadata

```bash
python prep_data.py
```

This creates `data/metadata.csv` as a skeleton for labeled audio.

### Generate synthetic training examples

```bash
python generate_synthetic_data.py
```

- Creates `data/X_train.npy` and `data/y_train.npy`
- Uses anchors defined in the script:
  - `data/raw_audio/test.wav` → label `4` (Happy/Stable)
  - `data/raw_audio/sad.wav` → label `1` (Sad/Depressed)

### Train the bridge network

```bash
python train_bridge.py
```

- Saves weights to `models/checkpoints/bridge_v1.pth`
- Uses a simple network mapping 1024-dimensional embeddings to 5 classes

### Test Emotion Embeddings

```bash
python train_emotion.py
```

- Loads `models/emotion2vec`
- Generates labels and embeddings for a test audio file
- Saves embeddings to `data/test_embedding.npy`

## Audio Emotion Inference

`therapist_bot.py` provides a separate classifier:

- Loads the Emotion2Vec encoder
- Loads the trained bridge model
- Predicts one of five emotional states:
  - `Anxious/Stressed`
  - `Sad/Depressed`
  - `Angry/Frustrated`
  - `Neutral/Calm`
  - `Happy/Stable`

## Frontend Notes

- The browser interface records microphone input and sends it as a `FormData` payload.
- `index.html` dynamically updates an emotion card based on the response.
- It also requests TTS audio from the backend after the assistant text is generated.

## Required Files and Directories

- `data/raw_audio/` — user recordings and sample anchor files.
- `data/X_train.npy` / `data/y_train.npy` — synthetic training dataset.
- `data/test_embedding.npy` — saved embedding example.
- `models/emotion2vec/` — Emotion2Vec model folder.
- `models/checkpoints/bridge_v1.pth` — trained bridge network weights.

## Dependencies

The project uses both the packages listed in `requirements.txt` and the following additional libraries:

- `flask`
- `requests`
- `python-dotenv`
- `ollama`
- `whisper`

## Known Caveats

- `app.py` expects an Inworld TTS key in `.env`.
- `brain.py` removes `*...*` and `[...]` markup from the streamed text.
- The frontend currently looks for a `METADATA|` prefix in the stream but the backend stream sends only cleaned text chunks. If you want live emotion metadata in the UI, that protocol must be aligned.
- The current training data is synthetic and limited to two anchor files, so the emotion classifier is only a prototype.

## Recommended Next Steps

1. Add more labeled audio samples to `data/raw_audio/`.
2. Expand `generate_synthetic_data.py` to support additional emotion anchors.
3. Improve the frontend stream metadata protocol.
4. Add real user session logging to `data/history.json` if needed.

## License

This project does not include a license file by default. Add `LICENSE` if you want to specify reuse terms.
