# SoulSync

SoulSync is a local, voice-first therapy-assistant prototype built with Flask. It records microphone input in the browser, transcribes speech with Whisper, detects emotion with a hybrid text+voice pipeline, streams an LLM response from Ollama, and synthesizes spoken output using the macOS `say` command.

This README documents the current code exactly as implemented in this repository.

## 1. What The App Does (Current Behavior)

1. Opens a single-page web UI at `/`.
2. Creates a fresh server-side session UUID on each page load.
3. Saves conversation state in `data/history/<session_id>.json`.
4. Records voice in browser with `MediaRecorder`.
5. Uploads audio to `/process_audio_stream`.
6. Saves uploaded audio to `data/raw_audio/<session_id>_input.wav`.
7. Transcribes with Whisper `base`.
8. Detects emotion in 2 phases:
   - Phase 1: large hardcoded keyword map.
   - Phase 2 fallback: Emotion2Vec embedding + PyTorch bridge classifier.
9. Streams the assistant text from Ollama (`llama3`) token-by-token.
10. Frontend requests `/get_audio` for TTS.
11. Generates an `.m4a` file with macOS `say -v Samantha -r 150`.
12. Plays generated audio from `static/audio/` in browser.

## 2. Important Reality Check (Code vs Legacy Config)

- `.env` is loaded (`load_dotenv()`), but `app.py` currently does not use `INWORLD_KEY`.
- TTS is local macOS `say`, not cloud TTS.
- `README_TRAINING.md` and some scripts represent a training pipeline that is partially separate from the live web path.
- The web path uses `brain.detect_emotion_hybrid(...)` (keyword first, neural fallback).

## 3. Repository Layout (All Project-Owned Files)

### 3.1 Core Application

- `app.py`
  - Flask app entrypoint.
  - Creates directories at startup:
    - `static/audio`
    - `data/raw_audio`
    - `data/history`
  - Routes:
    - `GET /`
    - `POST /process_audio_stream`
    - `POST /get_audio`
- `brain.py`
  - `TherapistBrain` orchestrator:
    - Whisper STT
    - emotion detection
    - conversation history read/write
    - Ollama streaming generation

### 3.2 Training + Voice Emotion Pipeline

- `train_emotion.py`
  - `EmotionProcessor` wrapper over FunASR `AutoModel` with local `models/emotion2vec`.
- `generate_synthetic_data.py`
  - Builds synthetic `X_train.npy` / `y_train.npy` from anchor audio embeddings.
- `train_bridge.py`
  - Defines and trains `TherapistBridge` (1024 -> 256 -> 5 classes).
- `therapist_bot.py`
  - Offline inference wrapper using `EmotionProcessor` + trained bridge checkpoint.

### 3.3 Utility Scripts

- `transcribe.py`
  - Standalone Whisper transcription check.
- `prep_data.py`
  - Creates empty CSV scaffold: `data/metadata.csv`.
- `how_to_open.txt`
  - Quick run notes:
    - `source venv/bin/activate`
    - `python3 app.py`

### 3.4 Frontend + Static

- `templates/index.html`
  - Tailwind UI, recording logic, stream parsing, TTS playback.
- `static/speech.aiff`
  - Static audio asset.
- `static/audio/*`
  - Generated local TTS outputs.

### 3.5 Data + Models

- `data/history/*.json`
  - Per-session transcript files.
- `data/history.json`
  - Additional history artifact.
- `data/raw_audio/*.wav`
  - Captured and sample training audio.
- `data/test_embedding.npy`
  - Embedding dump from `train_emotion.py` script run mode.
- `data/X_train.npy`, `data/y_train.npy`
  - Synthetic bridge training arrays.
- `data/vector_vault/`
  - Chroma DB artifact files (`chroma.sqlite3` + binary index).
- `models/emotion2vec/`
  - Local Emotion2Vec model directory:
    - `config.yaml`
    - `configuration.json`
    - `model.pt` (Git LFS tracked)
    - `tokens.txt`
- `models/checkpoints/bridge_v1.pth`
  - Trained bridge checkpoint.

### 3.6 Project Config / Metadata

- `requirements.txt`
- `.gitignore`
- `.gitattributes`
- `.env` (local, ignored)

### 3.7 Generated / Environment Artifacts In Repo Folder

- `__pycache__/...`
- `.DS_Store` files

## 4. Current Data Snapshot In This Workspace

As of this update:

- `data/history/`: 31 JSON files
- `data/raw_audio/`: 40 WAV files
- `static/audio/`: 36 generated audio files

These counts are repository-state facts and may grow during use.

## 5. Backend Route Contracts

### 5.1 `GET /`

Behavior:

1. Generates a new UUID and stores it in `session['id']`.
2. Creates `data/history/<session_id>.json` with initial payload:
   - `session_id`
   - `history` seeded with one `system` message (`brain.system_prompt`)
3. Renders `templates/index.html`.

Notes:

- Every page refresh creates a new session file.
- `app.secret_key = os.urandom(24)` changes every server restart (existing cookies become invalid).

### 5.2 `POST /process_audio_stream`

Input:

- `multipart/form-data` with file key `audio`.

Behavior:

1. Returns `400` if no audio key.
2. Saves upload to `data/raw_audio/<session_id>_input.wav`.
3. Calls:
   - `brain.transcribe_audio(...)`
   - `brain.detect_emotion_hybrid(user_text, audio_path)`
4. Starts plain-text stream response.
5. First stream segment is metadata in this exact format:
   - `METADATA|<transcribed_text>|<detected_emotion>|`
6. Remaining stream segments are assistant text chunks from `brain.generate_streaming_response(...)`.

Output:

- `text/plain` streaming body.

### 5.3 `POST /get_audio`

Input JSON:

- `text` (assistant text)

Behavior:

1. Sanitizes text by removing `*`, `[`, `]`, and replacing `"` with `'`.
2. Creates filename `luna_<uuid>.m4a`.
3. Runs local command:

```bash
say -v Samantha -r 150 -o static/audio/<filename>.m4a "<text>"
```

4. Returns JSON `{"audio_url": "/static/audio/<filename>"}` on success.
5. Returns `500` with error JSON on failure.

Notes:

- Requires macOS `say` command.
- Voice is hardcoded to `Samantha`.

## 6. TherapistBrain Deep Dive (`brain.py`)

## 6.1 Initialization

- Selects device:
  - `mps` if available
  - otherwise `cpu`
- Loads Whisper model: `whisper.load_model("base")`
- Initializes:
  - `EmotionProcessor()`
  - `TherapistBridge(input_dim=1024, num_classes=5)`
- Loads bridge checkpoint if present:
  - `models/checkpoints/bridge_v1.pth`
- Sets Ollama model name: `llama3`
- Defines long-form therapist system prompt emphasizing calm, concise responses.

## 6.2 STT

`transcribe_audio(audio_path)`:

- Returns `""` if file path does not exist.
- Calls Whisper with `fp16=False`.
- Returns stripped transcript.

## 6.3 Emotion Detection (Hybrid)

`detect_emotion_hybrid(text, audio_path)`:

1. Lowercases text.
2. Scans extensive keyword lists for:
   - `Sad`
   - `Anxious`
   - `Angry`
   - `Happy`
3. If keyword hit: return that category immediately.
4. Otherwise, neural fallback:
   - gets embedding from `EmotionProcessor`
   - converts to tensor shape `(1, 1024)`
   - bridge forward pass
   - class mapping:
     - `0 -> Anxious`
     - `1 -> Sad`
     - `2 -> Angry`
     - `3 -> Neutral`
     - `4 -> Happy`
5. Any exception in fallback returns `Neutral`.

## 6.4 History Persistence

- `_get_history(session_id)` loads `data/history/<session_id>.json` if present; else returns default system-only history.
- `_save_history(session_id, history)` writes JSON with `session_id` and full `history` array.

## 6.5 Streaming LLM Response

`generate_streaming_response(user_input, session_id)`:

1. If empty input, yields fallback sentence.
2. Loads history and appends current user message.
3. Calls `ollama.chat(..., stream=True)`.
4. For each chunk:
   - extracts `chunk['message']['content']`
   - strips stage-direction-like patterns via regex
   - yields cleaned text incrementally
5. Appends final assistant message to history and saves.

## 7. Frontend Behavior (`templates/index.html`)

1. Uses Tailwind via CDN.
2. Loads Google Font `Inter`.
3. Record button toggles between:
   - start recording
   - stop and process
4. Audio upload:
   - packs blob as `audio/wav`
   - filename `session.wav`
5. Reads stream via `ReadableStream` reader.
6. Parses metadata prefix `METADATA|...|...|`.
7. Updates emotion chip + card color theme by emotion label substring.
8. Aggregates full assistant text.
9. Calls `/get_audio` with text and detected emotion.
10. Plays returned audio with cache buster `?t=<timestamp>`.
11. Shows status labels: `Ready`, `Listening...`, `Processing...`, `Error`.

## 8. Training Pipeline Details

### 8.1 `train_emotion.py`

- Loads FunASR `AutoModel` from `models/emotion2vec`.
- `get_results(audio_path)` returns:
  - `labels_dict`: `{label: score}`
  - `embeddings`: NumPy vector from `res[0]['feats']`
- Script mode checks `data/raw_audio/test.wav`, prints report, saves `data/test_embedding.npy`.

### 8.2 `generate_synthetic_data.py`

- Anchor label mapping (hardcoded):
  - `data/raw_audio/test.wav -> 4`
  - `data/raw_audio/sad.wav -> 1`
- For each anchor, creates 100 noisy embedding variants using Gaussian noise `N(0, 0.02)`.
- Saves arrays:
  - `data/X_train.npy`
  - `data/y_train.npy`

### 8.3 `train_bridge.py`

- `TherapistBridge` architecture:
  - `Linear(1024, 256)`
  - `ReLU`
  - `Dropout(0.2)`
  - `Linear(256, 5)`
  - `LogSoftmax(dim=1)`
- Training:
  - full-batch over loaded arrays
  - optimizer: `Adam(lr=0.001)`
  - loss: `NLLLoss`
  - 50 epochs
  - logs every 10 epochs
- Saves checkpoint to `models/checkpoints/bridge_v1.pth`.

### 8.4 `therapist_bot.py`

- Loads `EmotionProcessor` + trained bridge checkpoint.
- `analyze_voice(audio_path)` returns one of:
  - `Anxious/Stressed`
  - `Sad/Depressed`
  - `Angry/Frustrated`
  - `Neutral/Calm`
  - `Happy/Stable`

### 8.5 `transcribe.py`

- Standalone Whisper check for `data/raw_audio/sad.wav`.

### 8.6 `prep_data.py`

- Creates empty `data/metadata.csv` with columns:
  - `file_path`
  - `label`

## 9. Environment Requirements

## 9.1 OS

- macOS is required for current TTS path (`say`).

## 9.2 Python

- A modern Python 3 environment (virtualenv recommended).

## 9.3 External Runtime Dependencies

- Ollama installed and running locally.
- Ollama model downloaded:

```bash
ollama pull llama3
```

- Local Emotion2Vec files in `models/emotion2vec/`.

## 9.4 Optional / Legacy `.env`

- `.env` is loaded.
- `INWORLD_KEY` exists in local environment file but is not used by the current `app.py` path.

## 10. Setup And Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

- `http://127.0.0.1:5000`

## 11. Script Commands

```bash
python prep_data.py
python train_emotion.py
python generate_synthetic_data.py
python train_bridge.py
python therapist_bot.py
python transcribe.py
```

## 12. Required And Generated Files

## 12.1 Must Exist (for full pipeline)

- `models/emotion2vec/config.yaml`
- `models/emotion2vec/configuration.json`
- `models/emotion2vec/model.pt`
- `models/emotion2vec/tokens.txt`
- `data/raw_audio/test.wav` (for default training scripts)
- `data/raw_audio/sad.wav` (for default training scripts)

## 12.2 Auto-Created During App Runtime

- `data/history/<session_id>.json`
- `data/raw_audio/<session_id>_input.wav`
- `static/audio/luna_<uuid>.m4a`

## 12.3 Auto-Created During Training Scripts

- `data/test_embedding.npy`
- `data/X_train.npy`
- `data/y_train.npy`
- `models/checkpoints/bridge_v1.pth`

## 13. Known Limitations

1. Frontend does not robustly guard malformed metadata chunks.
2. Keyword emotion matching uses substring logic and may over-trigger on partial words.
3. Neural fallback catches all exceptions broadly and returns `Neutral`, hiding root causes.
4. Training data generator defaults to only two classes (labels 1 and 4).
5. No train/validation split or evaluation metrics in bridge training.
6. No authentication/rate limiting in web routes.
7. Session history files can grow unbounded over time.

## 14. Troubleshooting

- No response stream:
  - verify Ollama service is running and `llama3` is available.
- No audio output:
  - ensure running on macOS with `say` command available.
- Neutral emotion always returned:
  - verify `models/checkpoints/bridge_v1.pth` exists and Emotion2Vec model files are valid.
- Empty transcript:
  - inspect saved wav in `data/raw_audio/` and test with `python transcribe.py`.
- Training fails at load stage:
  - regenerate training arrays with `python generate_synthetic_data.py`.

## 15. Security And Privacy Notes

1. Audio and conversation history are written to disk in plaintext project folders.
2. `.env` should remain uncommitted (already listed in `.gitignore`).
3. This repository currently contains generated artifacts and local history files; sanitize before sharing publicly.

## 16. Git/Storage Notes

- `.gitattributes` marks `models/emotion2vec/model.pt` as Git LFS managed.
- `.gitignore` currently ignores:
  - `.env`
  - `venv/`
  - `__pycache__/`
  - `*.wav`
  - `*.mp3`
  - `history/`

Note: `.gitignore` entry `history/` does not match `data/history/` directly.
