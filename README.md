# SoulSync

SoulSync is a local, voice-first therapeutic conversation assistant built with Flask. It records microphone audio in the browser, transcribes speech with Whisper, detects emotion with a hybrid stack (keywords + neural ensemble), streams an LLM reply from Ollama, and synthesizes local speech output on macOS.

This README is a deep, implementation-accurate snapshot of the project as currently present in this workspace.

## 1) Project At A Glance

Current runtime pipeline:

1. User opens the web app at GET /.
2. Backend creates a unique UUID session and initializes session history in data/history/<session_id>.json.
3. Browser records voice via MediaRecorder.
4. Audio is posted to POST /process_audio_stream.
5. Backend stores upload as data/raw_audio/<session_id>_input.wav.
6. Whisper base model transcribes the audio.
7. Emotion detection runs with:
   - silence/energy check
   - keyword override
   - neural ensemble fallback (fine-tuned Wav2Vec2 + Emotion2Vec)
8. Ollama streams response tokens from model llama3.
9. Frontend requests speech rendering via POST /get_audio.
10. Backend uses macOS say with Samantha voice to synthesize an m4a file under static/audio.
11. Frontend plays generated audio and updates UI state.

## 2) Current Repository Reality

Important: repository has evolved significantly from older bridge-only docs.

Current codebase facts:

- Active Python source files at project root:
  - app.py
  - brain.py
  - train_emotion.py
  - transcribe.py
- Files referenced in older docs such as train_bridge.py, therapist_bot.py, generate_synthetic_data.py, and prep_data.py are not present in this workspace.
- The active emotion classifier in runtime is a fine-tuned Wav2Vec2 model at custom_model/final_emotion_model, not a separate MLP bridge checkpoint.
- Emotion2Vec is still used (models/emotion2vec) as part of an ensemble fallback.

## 3) High-Level Architecture

### 3.1 Runtime Components

- Flask API server: app.py
- Core inference orchestration: brain.py
- Frontend: templates/index.html
- Speech-to-text: openai-whisper (base)
- LLM generation: ollama chat stream (model llama3)
- Emotion subsystem:
  - keyword map
  - fine-tuned Wav2Vec2ForSequenceClassification
  - Emotion2Vec via FunASR AutoModel
- TTS output:
  - local command line synthesizer say
  - voice Samantha
  - output files in static/audio

### 3.2 Data And Model Layers

- Session memory persisted per conversation as JSON in data/history.
- User raw recordings written to data/raw_audio.
- TTS render artifacts written to static/audio.
- Fine-tuned classifier weights in custom_model/final_emotion_model/model.safetensors.
- Emotion2Vec base local model in models/emotion2vec/model.pt (LFS-tracked).
- Vector DB artifacts present in data/vector_vault (Chroma SQLite + index binaries).

## 4) File-By-File Technical Breakdown

### 4.1 app.py

Responsibilities:

- Loads environment variables via python-dotenv load_dotenv().
- Initializes Flask app and random secret key: os.urandom(24).
- Ensures runtime directories exist:
  - static/audio
  - data/raw_audio
  - data/history
- Instantiates TherapistBrain once at startup.

Routes:

1. GET /
   - creates session id
   - writes initial history payload:
     - session_id
     - history array containing the system prompt
   - renders templates/index.html

2. POST /process_audio_stream
   - expects multipart file key audio
   - saves upload as data/raw_audio/<session_id>_input.wav
   - transcribes via brain.transcribe_audio
   - detects emotion via brain.detect_emotion_hybrid
   - streams plain-text response where first emitted block is metadata:
     - METADATA|<user_text>|<detected_emotion>|
   - then streams assistant chunks from brain.generate_streaming_response

3. POST /get_audio
   - reads JSON body with text
   - sanitizes characters (*, [, ], and double quote)
   - writes output filename Samantha_<uuid>.m4a (actual lowercase prefix in code: samantha_)
   - executes:

```bash
say -v Samantha -r 150 -o static/audio/<file>.m4a "<text>"
```

   - returns JSON audio_url on success
   - returns JSON error + HTTP 500 on failure

Server launch behavior:

```bash
python app.py
```

Runs Flask on port 5000 with debug=False and threaded=True.

### 4.2 brain.py

Core class: TherapistBrain

Initialization:

- Device selection: mps if available else cpu.
- Loads Whisper base model.
- Loads fine-tuned Wav2Vec2 classifier and feature extractor from:
  - custom_model/final_emotion_model
- Loads Emotion2Vec wrapper from train_emotion.py.
- Sets emotion id map:
  - 0 Happy
  - 1 Sad
  - 2 Angry
  - 3 Neutral
  - 4 Anxious
- Defines therapist prompt persona (Samantha).
- Defines large keyword dictionaries for Sad, Anxious, Angry, Happy.

Method behavior:

1. transcribe_audio(audio_path)
   - returns empty string if missing file
   - Whisper transcribe with fp16=False

2. detect_emotion_hybrid(text, audio_path)
   - Stage A: silence gate using librosa RMS; returns Neutral for low-energy input
   - Stage B: keyword check on lowercased transcript
   - Stage C: neural ensemble fallback
     - Wav2Vec2 probability vector from audio features
     - Emotion2Vec label scores projected into fixed class order
     - weighted fusion:

$$
P_{final} = 0.6 \cdot P_{wav2vec2} + 0.4 \cdot P_{emotion2vec}
$$

     - argmax picks final class
   - any exception in neural stage returns Neutral

3. _get_history(session_id)
   - reads data/history/<session_id>.json if available
   - otherwise returns single system message

4. _save_history(session_id, history)
   - writes full JSON history object with session_id and history

5. generate_streaming_response(user_input, session_id, detected_emotion=None)
   - handles empty/very-short quiet input with fallback greeting
   - formats user content with tone tag:
     - [User Tone: <emotion>] <text>
   - calls ollama.chat(..., stream=True)
   - strips stage/action patterns from streamed chunks using regex
   - yields cleaned chunks incrementally
   - appends assistant reply to persistent session history

### 4.3 train_emotion.py

Purpose:

- Thin local wrapper around FunASR Emotion2Vec model.

Key behaviors:

- Uses device mps/cpu.
- Loads local model folder models/emotion2vec with disable_update=True for offline use.
- get_results(audio_path) returns tuple:
  - labels_dict: label -> score
  - embeddings: numpy array of feats
- If file missing, returns (None, None).

Script mode behavior:

- test file expected: data/raw_audio/test.wav
- prints top emotion and confidence
- prints embedding shape and preview
- saves embeddings to data/test_embedding.npy

### 4.4 transcribe.py

Purpose:

- Standalone smoke test wrapper around Whisper.

Behavior:

- Loads Whisper base on mps/cpu.
- transcribe(audio_path) returns cleaned transcript string.
- Script mode tests data/raw_audio/sad.wav.

### 4.5 templates/index.html

Frontend behavior details:

- Tailwind CDN + custom CSS glassmorphism UI.
- Uses Inter font from Google Fonts.
- Main control: single record button toggles start/stop.
- MediaRecorder captures browser microphone stream.
- Sends Blob as audio/wav to backend.
- Reads streaming response via ReadableStream reader.
- Parses metadata header from first response chunk.
- Updates emotion badge and dynamic card style by emotion category.
- Requests synthesized audio from /get_audio.
- Plays returned audio URL with cache-busting timestamp query.

UI state labels:

- Ready
- Listening...
- Processing...
- Error

Known API mismatch:

- Frontend sends { text, emotion } to /get_audio.
- Backend currently uses text only.

### 4.6 requirements.txt

Declared dependencies:

- torch
- numpy
- pandas
- librosa
- soundfile
- openai-whisper
- funasr
- modelscope
- Flask
- python-dotenv
- ollama
- torchvision
- torchaudio

Note:

- brain.py imports transformers (Wav2Vec2 classes), but transformers is not listed in requirements.txt. A fresh environment may require manual install of transformers.

### 4.7 README_TRAINING.md

Contains a training-oriented write-up, but parts are from an older bridge-based architecture and should be treated as historical context unless synchronized with current files.

### 4.8 how_to_open.txt

Current startup notes:

```bash
source venv/bin/activate
python3 app.py
```

## 5) Models And Training Assets

### 5.1 Fine-Tuned Emotion Classifier

Path: custom_model/final_emotion_model

Files:

- config.json
- preprocessor_config.json
- model.safetensors

Important config points from config.json:

- architecture: Wav2Vec2ForSequenceClassification
- num labels: 5 (happy, sad, angry, neutral, anxious)
- hidden size: 768
- transformers_version: 5.5.4

Important preprocessor points:

- sampling_rate: 16000
- do_normalize: true
- return_attention_mask: false

### 5.2 Emotion2Vec Base Model

Path: models/emotion2vec

Files:

- config.yaml
- configuration.json
- model.pt
- tokens.txt

Model details in local metadata:

- task: emotion-recognition
- framework: pytorch
- model_name_in_hub.ms: iic/emotion2vec_base
- embed_dim (from config.yaml): 1024
- supported modality: AUDIO

### 5.3 Training Notebook

Path: custom_model/train.ipynb

Notebook workflow summary:

1. Installs seaborn.
2. Loads dataset and maps emotions to 5-class target_map.
3. Uses librosa for manual loading/resampling at 16kHz.
4. Uses Hugging Face Dataset + Wav2Vec2FeatureExtractor preprocessing.
5. Fine-tunes facebook/wav2vec2-base-960h with Trainer.
6. Uses train/validation split with stratification.
7. Evaluates accuracy and plots confusion matrix + training curves.
8. Saves final model to custom_model/final_emotion_model.
9. Includes standalone local inference test cells.

## 6) Dataset Inventory

Primary labeled dataset root:

- custom_model/Emotions

Observed file counts by class directory:

- Angry: 2167
- Disgusted: 1863
- Fearful: 2047
- Happy: 2167
- Neutral: 1795
- Sad: 2167
- Suprised: 592

Total files under custom_model directory: 12805

Notes:

- The folder name Suprised is intentionally spelled as present in workspace.
- Training notebook maps a five-emotion target subset for the deployed classifier.

## 7) Runtime Data Artifacts

### 7.1 data directory

Contains:

- history/ (55 JSON session files currently present)
- raw_audio/ (4 wav files currently present)
- test_embedding.npy
- vector_vault/

### 7.2 Session history schema

Each file in data/history follows this shape:

```json
{
  "session_id": "<uuid>",
  "history": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 7.3 Vector vault contents

data/vector_vault contains:

- chroma.sqlite3
- d6712cb2-30b1-4829-af87-8fe9b169ae7d/data_level0.bin
- d6712cb2-30b1-4829-af87-8fe9b169ae7d/header.bin
- d6712cb2-30b1-4829-af87-8fe9b169ae7d/length.bin
- d6712cb2-30b1-4829-af87-8fe9b169ae7d/link_lists.bin

### 7.4 static directory

Contains:

- static/speech.aiff
- static/audio/ with generated files (62 currently present)

Observed filenames include legacy prefixes from older experiments (bark_, luna_) and current Samantha outputs (samantha_*.m4a).

## 8) Operational Setup

### 8.1 Prerequisites

- macOS (required for current say-based TTS path)
- Python virtual environment
- Ollama installed and running locally
- Local Ollama model available:

```bash
ollama pull llama3
```

### 8.2 Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install transformers
```

### 8.3 Run App

```bash
source venv/bin/activate
python app.py
```

Open http://127.0.0.1:5000

## 9) API Contracts

### 9.1 POST /process_audio_stream

Request:

- multipart/form-data
- required file field: audio

Response:

- content-type text/plain stream
- first payload block:
  - METADATA|<transcript>|<emotion>|
- remainder:
  - assistant text chunks

### 9.2 POST /get_audio

Request JSON:

```json
{
  "text": "assistant response text",
  "emotion": "optional/ignored by backend currently"
}
```

Success response:

```json
{
  "audio_url": "/static/audio/samantha_<id>.m4a"
}
```

Failure response:

```json
{
  "error": "..."
}
```

## 10) Known Gaps, Risks, And Behavioral Edge Cases

1. requirements.txt does not currently include transformers although runtime imports it.
2. app.secret_key is regenerated on each restart, invalidating old session cookies.
3. /get_audio is macOS-specific because it depends on say.
4. /get_audio sanitizes text but still executes system TTS command; keep text lengths practical.
5. Any exception in neural emotion stage falls back to Neutral.
6. Legacy artifacts (history prompts, old audio prefixes) coexist with current Samantha runtime.
7. .gitignore excludes history/ but app writes data/history/; generated JSON session logs remain in repo workspace unless separately cleaned.
8. Repository is large (over 71k files), primarily due to dataset and environment artifacts.

## 11) Full Top-Level Inventory (Current Workspace)

- .DS_Store
- .env
- .gitattributes
- .gitignore
- app.py
- brain.py
- custom_model/
- data/
- how_to_open.txt
- models/
- README.md
- README_TRAINING.md
- requirements.txt
- static/
- templates/
- train_emotion.py
- transcribe.py
- venv/
- __pycache__/

## 12) Evolution Notes (For Maintainers)

The project contains evidence of multiple development phases:

- Early/alternate persona and TTS naming (Luna, bark) in artifacts.
- Current production code persona and TTS pathway: Samantha + macOS say.
- Old bridge-model documentation remains in README_TRAINING.md, while current active implementation in brain.py uses Wav2Vec2 + Emotion2Vec ensemble.

When making future changes, keep docs synchronized with app.py and brain.py first, then update training docs and artifact notes.
