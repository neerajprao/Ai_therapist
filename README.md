# SoulSync

SoulSync is a local, voice-first therapy-assistant prototype built with Flask. The current runtime records microphone input in the browser, transcribes speech with Whisper, detects emotion with a hybrid text-plus-voice pipeline, streams an LLM response from Ollama, and synthesizes spoken output with the macOS `say` command.

This README is written to match the repository exactly as it exists now, including the live web app, the offline training scripts, and the on-disk artifacts already present in the workspace.

## 1. Current Runtime Summary

The web app path is:

1. Open `GET /`.
2. A fresh server-side UUID session is created and stored in `session['id']`.
3. A new conversation file is written to `data/history/<session_id>.json` with the system prompt as the first message.
4. The browser records microphone audio with `MediaRecorder`.
5. The audio blob is posted to `POST /process_audio_stream` as `audio`.
6. The upload is saved to `data/raw_audio/<session_id>_input.wav`.
7. Whisper transcribes the audio with the `base` model.
8. Emotion detection runs in two phases:
   - Phase 1: keyword lookup over large hardcoded word lists.
   - Phase 2: Emotion2Vec embedding extraction plus the trained bridge classifier.
9. The assistant response is streamed token-by-token from Ollama `llama3`.
10. The client collects the full assistant text and calls `POST /get_audio`.
11. The backend synthesizes speech locally with the macOS `say` command using the `Samantha` voice.
12. The generated file is served from `static/audio/` and played in the browser.

## 2. Important Behavioral Notes

- The app loads `.env` with `python-dotenv`, but the current `app.py` code does not use `INWORLD_KEY` or any other environment value for runtime behavior.
- `app.secret_key = os.urandom(24)` changes every server restart, so old session cookies stop working after a restart.
- The current TTS path is local and macOS-specific. It does not use a cloud TTS provider.
- `POST /get_audio` ignores the submitted `emotion` field. The frontend sends it, but the backend only uses `text`.
- The live emotion pipeline is not purely neural. `brain.detect_emotion_hybrid(...)` returns a keyword-based label first whenever a keyword matches.
- The repository contains both live app code and an offline training stack. They are related, but they are not the same execution path.

## 3. Repository Structure

### 3.1 Core Runtime

- `app.py`
  - Flask entrypoint.
  - Creates runtime directories at startup:
    - `static/audio`
    - `data/raw_audio`
    - `data/history`
  - Defines the app routes:
    - `GET /`
    - `POST /process_audio_stream`
    - `POST /get_audio`
  - Runs the development server on port `5000` with `debug=False` and `threaded=True`.
- `brain.py`
  - `TherapistBrain` orchestrator.
  - Handles Whisper transcription, hybrid emotion detection, history persistence, and streamed Ollama generation.

### 3.2 Training And Offline Inference

- `train_emotion.py`
  - Wraps local Emotion2Vec via FunASR `AutoModel`.
  - Extracts both labels and embeddings from voice audio.
- `generate_synthetic_data.py`
  - Builds synthetic NumPy training arrays from anchor audio files.
- `train_bridge.py`
  - Defines and trains the small bridge classifier that maps embeddings to five emotion classes.
- `therapist_bot.py`
  - Offline inference wrapper that loads the Emotion2Vec encoder and the trained bridge checkpoint.
- `transcribe.py`
  - Standalone Whisper transcription check for a sample file.
- `prep_data.py`
  - Creates an empty metadata CSV scaffold.

### 3.3 Frontend

- `templates/index.html`
  - Single-page UI.
  - Recording controls, streaming response handling, emotion theme updates, and audio playback.

### 3.4 Data And Model Artifacts

- `data/history.json`
  - Additional history artifact currently present in the repo.
- `data/history/`
  - Per-session conversation files written by the app.
- `data/raw_audio/`
  - Uploaded and sample audio files.
- `data/X_train.npy`
  - Synthetic bridge input features.
- `data/y_train.npy`
  - Synthetic bridge labels.
- `data/test_embedding.npy`
  - Saved Emotion2Vec embedding from `train_emotion.py` script mode.
- `data/vector_vault/`
  - Chroma database artifact currently checked into the workspace.
- `models/emotion2vec/`
  - Local Emotion2Vec model directory.
- `models/checkpoints/bridge_v1.pth`
  - Trained bridge checkpoint.
- `static/speech.aiff`
  - Static audio asset present in the repo.
- `static/audio/`
  - Generated TTS outputs.

### 3.5 Project Metadata And Environment Files

- `requirements.txt`
- `.gitignore`
- `.gitattributes`
- `.env`
- `README_TRAINING.md`
- `how_to_open.txt`
- `venv/`
- `__pycache__/`
- `.DS_Store`

## 4. Runtime File-By-File Details

### 4.1 `app.py`

`app.py` is the Flask server.

Behavior:

1. Calls `load_dotenv()` at startup.
2. Creates the `static/audio`, `data/raw_audio`, and `data/history` directories if they do not exist.
3. Imports `TherapistBrain` and creates one `brain` instance.
4. On `GET /`:
   - generates a new UUID
   - stores it in `session['id']`
   - writes `data/history/<session_id>.json`
   - renders `templates/index.html`
5. On `POST /process_audio_stream`:
   - expects multipart form data with the `audio` file key
   - saves the file to `data/raw_audio/<session_id>_input.wav`
   - transcribes it with Whisper
   - detects emotion with the hybrid pipeline
   - streams metadata first, then assistant text chunks
6. On `POST /get_audio`:
   - reads JSON body text from `request.json`
   - sanitizes the text by removing `*`, `[` and `]`, and replacing `"` with `'`
   - writes an `.m4a` file in `static/audio/`
   - invokes macOS `say -v Samantha -r 150 -o <file>`
   - returns `{"audio_url": "/static/audio/<filename>"}` on success
   - returns HTTP 500 with JSON error on failure

Stream format from `/process_audio_stream`:

- The first chunk is always:
  - `METADATA|<transcribed_text>|<detected_emotion>|`
- The rest of the stream is the generated assistant text.

### 4.2 `brain.py`

`brain.py` contains the runtime logic behind the web app.

Initialization:

- Selects `mps` if available, otherwise `cpu`.
- Loads Whisper `base`.
- Instantiates `EmotionProcessor()` from `train_emotion.py`.
- Instantiates `TherapistBridge(input_dim=1024, num_classes=5)` from `train_bridge.py`.
- Loads `models/checkpoints/bridge_v1.pth` if it exists.
- Sets the Ollama model name to `llama3`.
- Defines the system prompt used for the therapist persona.
- Defines the keyword map for `Sad`, `Anxious`, `Angry`, and `Happy`.

Methods:

- `transcribe_audio(audio_path)`
  - returns an empty string if the file does not exist
  - calls Whisper with `fp16=False`
  - returns the stripped transcript
- `detect_emotion_hybrid(text, audio_path)`
  - lowercases the text
  - checks the keyword map first
  - if no keyword matches, extracts an Emotion2Vec embedding and runs the bridge model
  - maps bridge output classes as follows:
    - `0 -> Anxious`
    - `1 -> Sad`
    - `2 -> Angry`
    - `3 -> Neutral`
    - `4 -> Happy`
  - returns `Neutral` on any exception in the fallback path
- `_get_history(session_id)`
  - loads `data/history/<session_id>.json` if present
  - otherwise returns a default history containing only the system prompt
- `_save_history(session_id, history)`
  - writes the full history back to disk as JSON
- `generate_streaming_response(user_input, session_id)`
  - yields a fallback line if `user_input` is empty
  - loads history and appends the user message
  - calls `ollama.chat(..., stream=True)`
  - strips bracketed or starred stage-direction-like text from each streamed chunk
  - yields cleaned text incrementally
  - saves the final assistant message to history after streaming completes

### 4.3 `templates/index.html`

The frontend is a single HTML page with Tailwind CDN styling and an imported Google Font.

Behavior:

1. Shows a centered recording interface with a microphone button.
2. Uses the `MediaRecorder` API to capture microphone audio.
3. On stop, packages the recording as a blob labeled `audio/wav` and sends it to `/process_audio_stream`.
4. Reads the streamed response with a `ReadableStream` reader.
5. Parses the metadata prefix using `METADATA|...|...|`.
6. Updates the displayed transcript snippet and detected emotion.
7. Switches the emotion card theme based on the emotion label substring.
8. Collects the full assistant response text.
9. Posts that text to `/get_audio`.
10. Plays the returned audio file with a cache-busting timestamp query string.
11. Updates the status text through `Ready`, `Listening...`, `Processing...`, and `Error`.

Important frontend/backend mismatch:

- The frontend sends `{ text, emotion }` to `/get_audio`, but the backend only uses `text`.

### 4.4 `train_emotion.py`

`train_emotion.py` provides the Emotion2Vec wrapper.

Behavior:

- Selects `mps` if available, otherwise `cpu`.
- Loads the local model from `models/emotion2vec`.
- Uses `AutoModel(..., disable_update=True)` so the model stays local and offline.
- `get_results(audio_path)` returns a pair:
  - `labels_dict`: mapping from label string to score
  - `embeddings`: NumPy array of the feature vector
- If the audio file does not exist, it returns `(None, None)`.

Script mode (`python train_emotion.py`):

1. Reads `data/raw_audio/test.wav`.
2. Prints the top emotion and its confidence.
3. Prints the embedding shape and preview.
4. Saves the embedding to `data/test_embedding.npy`.

### 4.5 `generate_synthetic_data.py`

This script builds the synthetic training set for the bridge classifier.

Hardcoded anchors:

- `data/raw_audio/test.wav -> 4`
- `data/raw_audio/sad.wav -> 1`

Behavior:

1. Instantiates `EmotionProcessor`.
2. For each anchor file that exists, extracts its embedding.
3. Creates 100 noisy variants per file with Gaussian noise `N(0, 0.02)`.
4. Appends all samples to `X` and labels to `y`.
5. Saves:
   - `data/X_train.npy`
   - `data/y_train.npy`

Important consequence:

- The default synthetic dataset only contains two labels, even though the bridge output space has five classes.

### 4.6 `train_bridge.py`

This script trains the small MLP bridge.

Model:

- `TherapistBridge(input_dim=1024, num_classes=5)`
- Architecture:
  - `Linear(1024, 256)`
  - `ReLU`
  - `Dropout(0.2)`
  - `Linear(256, 5)`
  - `LogSoftmax(dim=1)`

Training:

- Loads `data/X_train.npy` and `data/y_train.npy`.
- Uses `mps` if available, otherwise `cpu`.
- Uses `Adam(lr=0.001)`.
- Uses `NLLLoss`.
- Trains for 50 epochs.
- Prints loss every 10 epochs.
- Saves the checkpoint to `models/checkpoints/bridge_v1.pth`.

Failure behavior:

- If `data/X_train.npy` does not exist, the script prints an error and returns without training.

### 4.7 `therapist_bot.py`

This is the offline inference wrapper.

Behavior:

1. Loads `EmotionProcessor`.
2. Loads `TherapistBridge` from `models/checkpoints/bridge_v1.pth`.
3. Converts the embedding to shape `(1, 1024)`.
4. Runs inference through the bridge.
5. Maps the prediction using `EMOTION_MAP`:
   - `0 -> Anxious/Stressed`
   - `1 -> Sad/Depressed`
   - `2 -> Angry/Frustrated`
   - `3 -> Neutral/Calm`
   - `4 -> Happy/Stable`

Important detail:

- Unlike `brain.py`, this script does not have a checkpoint existence guard, so a missing `bridge_v1.pth` will raise when it tries to load the weights.

### 4.8 `transcribe.py`

This is a standalone Whisper smoke test.

Behavior:

- Loads Whisper `base`.
- Moves the model to `mps` if available, otherwise `cpu`.
- Transcribes `data/raw_audio/sad.wav`.
- Prints the transcript.

### 4.9 `prep_data.py`

This script creates an empty metadata CSV scaffold.

Output:

- `data/metadata.csv`

Columns:

- `file_path`
- `label`

Current state:

- No other script in the repository reads this CSV yet.

## 5. Complete File Inventory

This repository currently contains the following project-owned files and directories:

- `.env`
  - Local environment variables loaded by `app.py`.
  - Present in the workspace but ignored by Git.
- `.gitignore`
  - Ignores `.env`, `venv/`, `__pycache__/`, `*.wav`, `*.mp3`, and `history/`.
  - Note: the app uses `data/history/`, not a root-level `history/` directory.
- `.gitattributes`
  - Marks `models/emotion2vec/model.pt` for Git LFS with `filter=lfs diff=lfs merge=lfs -text`.
- `.DS_Store`
  - macOS Finder metadata file; not part of the app.
- `README.md`
  - Main repository documentation.
- `README_TRAINING.md`
  - Training/offline pipeline documentation.
- `app.py`
  - Flask web app.
- `brain.py`
  - Live runtime brain/orchestrator.
- `generate_synthetic_data.py`
  - Synthetic bridge dataset generator.
- `how_to_open.txt`
  - Minimal startup note with activation and launch commands.
- `prep_data.py`
  - CSV scaffold generator.
- `requirements.txt`
  - Python dependencies.
- `therapist_bot.py`
  - Offline voice-state inference script.
- `train_bridge.py`
  - Bridge classifier trainer.
- `train_emotion.py`
  - Emotion2Vec wrapper and embedding extractor.
- `transcribe.py`
  - Standalone transcription test.
- `templates/index.html`
  - Web UI.
- `static/speech.aiff`
  - Static audio file.
- `static/audio/`
  - Generated speech output directory.
- `data/history.json`
  - Extra history artifact.
- `data/history/`
  - Per-session history JSON files.
- `data/raw_audio/`
  - Captured audio files and sample anchors.
- `data/test_embedding.npy`
  - Saved embedding from the Emotion2Vec script.
- `data/X_train.npy`
  - Synthetic training features.
- `data/y_train.npy`
  - Synthetic training labels.
- `data/vector_vault/`
  - Chroma DB artifacts.
- `models/checkpoints/bridge_v1.pth`
  - Trained bridge checkpoint.
- `models/emotion2vec/`
  - Local Emotion2Vec model files.
- `venv/`
  - Local virtual environment.
- `__pycache__/`
  - Python bytecode cache.

## 6. Data Snapshot Present In This Workspace

The workspace currently includes:

- a populated `data/history/` directory with many session JSON files
- a populated `data/raw_audio/` directory with sample and session audio files
- a populated `static/audio/` directory with generated TTS outputs
- a `data/vector_vault/` directory containing a Chroma SQLite database and index files

These are runtime artifacts, not source code, and they may grow as the app is used.

## 7. Setup And Run

### 7.1 Local Environment

Use a Python virtual environment and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 7.2 Required Runtime Dependencies

The live app expects all of the following to be available:

- macOS, because the TTS path uses `say`
- Ollama installed and running locally
- the `llama3` model pulled locally
- the local Emotion2Vec directory under `models/emotion2vec/`

Useful Ollama setup command:

```bash
ollama pull llama3
```

### 7.3 Start The App

```bash
python3 app.py
```

Then open:

- `http://127.0.0.1:5000`

## 8. Script Commands

```bash
python prep_data.py
python train_emotion.py
python generate_synthetic_data.py
python train_bridge.py
python therapist_bot.py
python transcribe.py
```

## 9. Current Limitations And Operational Risks

1. The default synthetic bridge dataset only covers two labels.
2. There is no train/validation/test split.
3. There is no reproducibility seed for the synthetic augmentation.
4. The neural fallback in `brain.py` uses a broad `except`, which can hide real failures.
5. The bridge classifier is trained full-batch on a tiny synthetic dataset.
6. The frontend sends an `emotion` field to `/get_audio`, but the backend ignores it.
7. The TTS implementation depends on the macOS `say` binary, so it is not portable as-is.
8. `app.secret_key` is regenerated on each server restart, so sessions do not survive restarts.
9. `models/emotion2vec/model.pt` is stored with Git LFS, so clones need LFS support to fetch it correctly.

## 10. Change Surface Worth Watching

If you change any of the following, update this README as well:

- the keyword lists or emotion labels in `brain.py`
- the model name in `brain.py`
- the bridge architecture in `train_bridge.py`
- the Emotion2Vec model path in `train_emotion.py`
- the TTS voice or command in `app.py`
- the frontend metadata parsing in `templates/index.html`
- the contents of `models/emotion2vec/`
- the checkpoint path in `models/checkpoints/`
