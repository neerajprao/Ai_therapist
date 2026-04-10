# SoulSync - AI Therapy Voice Assistant

SoulSync is a voice-first therapy assistant prototype with a Flask web app, local speech transcription, locally hosted LLM response generation, emotion-aware TTS playback, and a separate voice-emotion training pipeline.

This README is the complete project guide for setup, architecture, API behavior, data artifacts, and all scripts in this repository.

## 1) What the Project Does

- Records user speech in browser.
- Transcribes audio locally with Whisper.
- Detects a coarse text emotion (Sad, Anxious, Neutral).
- Streams therapist response text from Ollama (llama3) in real time.
- Converts final response into audio with Inworld TTS voice Luna.
- Stores per-session chat history in JSON files.
- Includes an independent training stack for voice emotion classification with Emotion2Vec + a PyTorch bridge model.

## 2) Runtime Architecture

### Web layer

- Backend: Flask app in app.py.
- Frontend: Single HTML page in templates/index.html using Tailwind CDN and browser MediaRecorder.
- Session behavior: each page refresh creates a new session ID and a fresh session history file.

### Inference layer (main app)

- STT: whisper base model (local).
- LLM: ollama chat API with model llama3 (local).
- TTS: Inworld cloud API, voiceId Luna, modelId inworld-tts-1.5-max.
- Text emotion detector: keyword rules in brain.py.

### Training layer (separate from live chat)

- Emotion embeddings: Emotion2Vec loaded from models/emotion2vec.
- Bridge model: TherapistBridge maps 1024-d embedding -> 5 emotion classes.
- Synthetic data generation: noisy embedding augmentation from anchor audio files.

## 3) Repository Map (Every File)

### Application entry points

- app.py: Flask app, routes, session/history bootstrap, stream response, TTS request, audio file output.
- brain.py: TherapistBrain class (Whisper load, transcription, emotion rules, Ollama streaming, history load/save).

### Emotion training and analysis

- train_emotion.py: EmotionProcessor around FunASR AutoModel for labels + embeddings.
- generate_synthetic_data.py: Builds X_train.npy and y_train.npy from anchor files + Gaussian noise.
- train_bridge.py: Defines and trains TherapistBridge network, saves bridge_v1.pth.
- therapist_bot.py: Loads EmotionProcessor + trained bridge and predicts one of 5 mapped emotions.

### Utilities

- transcribe.py: Standalone Whisper transcription test.
- prep_data.py: Creates empty data/metadata.csv with file_path,label columns.
- how_to_open.txt: minimal quick launch commands.

### Frontend + static

- templates/index.html: UI, recorder logic, streaming parser, emotion-theme update, TTS playback.
- static/speech.aiff: static audio asset.
- static/audio/: generated mp3 response files are saved here.

### Data + model artifacts

- data/history/: per-session JSON histories, one file per session UUID.
- data/history.json: additional history artifact.
- data/raw_audio/: recorded user inputs and training anchor audio.
- data/test_embedding.npy: saved embedding test output from train_emotion.py.
- data/X_train.npy, data/y_train.npy: synthetic training tensors saved as NumPy arrays.
- data/vector_vault/: Chroma vector store files (chroma.sqlite3 and index binaries).
- models/emotion2vec/: local Emotion2Vec files (config.yaml, configuration.json, model.pt, tokens.txt).
- models/checkpoints/bridge_v1.pth: trained bridge checkpoint.

### Environment/config/dev files

- requirements.txt: python dependencies.
- .gitignore: ignores .env, venv, __pycache__, wav/mp3, history folder.
- .env (not committed): expected to contain INWORLD_KEY.

## 4) End-to-End Request Flow

1. User opens /.
2. Server creates new session UUID and writes data/history/<session>.json initialized with system prompt.
3. Browser records microphone audio with MediaRecorder.
4. Frontend posts form-data audio file to /process_audio_stream.
5. Backend saves audio as data/raw_audio/<session>_input.wav.
6. brain.py transcribes user audio.
7. brain.py applies keyword emotion rules.
8. Streaming response starts with prefix: METADATA|<user_text>|<emotion>|.
9. Backend streams cleaned LLM chunks (stage directions removed).
10. Frontend accumulates full text and renders live UI.
11. Frontend posts full text + emotion to /get_audio.
12. Backend builds emotion-conditioned TTS prompt and calls Inworld API.
13. Backend saves decoded base64 audio to static/audio/luna_<uuid>.mp3.
14. Frontend plays returned audio URL and reveals full response text.

## 5) API Contract

### GET /

- Purpose: serve UI and initialize a new session/history file.
- Side effect: replaces session id on every refresh.

### POST /process_audio_stream

- Input: multipart form with key audio.
- Output: text/plain streaming response.
- First emitted token format: METADATA|user_text|emotion|.
- Remaining stream: assistant text chunks from Ollama.
- Error case: returns 400 with No audio when audio is missing.

### POST /get_audio

- Input JSON:
	- text: final assistant text.
	- emotion: detected emotion label.
- Behavior:
	- removes *, [ and ] from text.
	- speed fixed to 1.5.
	- default temperature 1.1.
	- Sad/Depressed/Grief -> prompt prepends [sad] [whispering] [sigh], temperature 1.4.
	- Anxious -> prompt prepends [soothing] [breathe], temperature 0.7.
	- otherwise -> prompt prepends [calm].
- Output JSON success: {"audio_url": "/static/audio/<file>.mp3"}.
- Output JSON failure: {"error": "..."} with status 500.

## 6) Core Classes and Behavior

### TherapistBrain (brain.py)

- Loads whisper base at initialization.
- Uses model_name = llama3 for Ollama chat.
- System prompt forces neutral, calm therapist behavior, concise 1-2 sentence outputs, and a gentle question/invitation.
- transcribe_audio:
	- returns empty string if file missing.
	- calls Whisper transcribe with fp16=False.
- detect_emotion keyword rules:
	- Sad if text has sad/empty/grief/hurts/lost.
	- Anxious if text has scared/anxious/worry/panic.
	- else Neutral.
- _get_history / _save_history:
	- reads/writes data/history/<session>.json.
- generate_streaming_response:
	- accepts pre_transcribed_text to avoid duplicate STT.
	- if no user input: yields fallback sentence and exits.
	- appends user message to history.
	- streams ollama.chat chunks.
	- strips *...* and [...]-style stage directions from chunks.
	- appends final assistant reply to history and persists.

### EmotionProcessor (train_emotion.py)

- Loads FunASR AutoModel from models/emotion2vec.
- Device: mps when available, else cpu.
- get_results(audio_path):
	- returns (None, None) if missing file.
	- returns labels dictionary and embedding array from generate(..., extract_embedding=True).

### TherapistBridge (train_bridge.py)

- Architecture: Linear(1024,256) -> ReLU -> Dropout(0.2) -> Linear(256,5) -> LogSoftmax(dim=1).

### TherapistBot (therapist_bot.py)

- Loads EmotionProcessor and bridge checkpoint models/checkpoints/bridge_v1.pth.
- analyze_voice:
	- obtains embeddings.
	- reshapes to batch dimension.
	- predicts class argmax.
	- maps class to human label.
- EMOTION_MAP:
	- 0 Anxious/Stressed
	- 1 Sad/Depressed
	- 2 Angry/Frustrated
	- 3 Neutral/Calm
	- 4 Happy/Stable

## 7) Frontend Behavior (templates/index.html)

- Uses MediaRecorder for audio capture.
- Record button toggles between recording and processing states.
- Sends blob as wav named session.wav to backend.
- Parses streamed metadata prefix before assistant text.
- Applies emotion-based theme classes:
	- Happy -> emerald
	- Sad -> purple
	- Angry -> red
	- Anxious -> amber
	- Neutral -> blue
- After text stream ends, requests TTS and plays returned audio URL.
- Adds cache-busting query param to audio URL with timestamp.
- UI status labels: Ready, Listening..., Processing..., Error.

## 8) Installation and Setup

### Prerequisites

- Python 3.8+
- Ollama installed locally
- llama3 model pulled in Ollama
- Inworld API credentials (base64 basic auth value)
- Emotion2Vec local model files in models/emotion2vec

### Setup

1. Create virtual environment.

	 python3 -m venv venv

2. Activate virtual environment.

	 source venv/bin/activate

3. Install dependencies.

	 pip install -r requirements.txt

4. Add .env in repo root.

	 INWORLD_KEY=your_base64_basic_auth_value

5. Ensure Ollama model is ready.

	 ollama pull llama3

## 9) Run the App

1. Activate env.

	 source venv/bin/activate

2. Start server.

	 python app.py

3. Open browser.

	 http://127.0.0.1:5000

## 10) Training and Testing Commands

- Create metadata skeleton:

	python prep_data.py

- Generate synthetic training data:

	python generate_synthetic_data.py

- Train bridge model:

	python train_bridge.py

- Run embedding test:

	python train_emotion.py

- Run bridge inference test:

	python therapist_bot.py

- Run transcription test:

	python transcribe.py

## 11) Data/Model File Expectations

- Required for app runtime:
	- .env with INWORLD_KEY
	- local Whisper model download (auto by package)
	- local Ollama llama3
- Required for voice-emotion pipeline:
	- models/emotion2vec/* files
	- data/raw_audio/test.wav and data/raw_audio/sad.wav (as currently scripted)
	- models/checkpoints/bridge_v1.pth for therapist_bot.py
- Auto-created at runtime:
	- data/history/*.json
	- data/raw_audio/<session>_input.wav
	- static/audio/luna_<uuid>.mp3

## 12) Dependencies (requirements.txt)

- torch
- torchvision
- torchaudio
- funasr
- modelscope
- librosa
- numpy
- pandas
- soundfile
- flask
- requests
- python-dotenv
- ollama
- whisper

## 13) Platform Notes

- Code prefers Apple Metal Performance Shaders (mps) when available.
- Falls back to cpu automatically.
- Designed and tested on macOS workflow.

## 14) Known Limitations

- Text emotion detector is simple keyword matching, not a learned NLP classifier.
- Voice emotion model is trained on synthetic augmentation from two anchor classes in current script defaults.
- No validation split/metrics in bridge training loop.
- Session is reset on page refresh by design.
- If Inworld key is missing/invalid, TTS endpoint fails.
- If Ollama or llama3 is unavailable, streaming response fails.

## 15) Troubleshooting

- No audio captured:
	- verify browser microphone permission and secure context support.
- No streamed response:
	- confirm Ollama service is running and llama3 exists.
- TTS error JSON from /get_audio:
	- validate INWORLD_KEY format and quota.
- Missing model errors:
	- ensure models/emotion2vec files and bridge_v1.pth are present.
- Empty transcription:
	- confirm input wav exists and contains intelligible speech.

## 16) Quick Start (Minimal)

1. source venv/bin/activate
2. pip install -r requirements.txt
3. set INWORLD_KEY in .env
4. ollama pull llama3
5. python app.py

## 17) License

No license file is currently included. Add a LICENSE file if you want explicit reuse terms.
