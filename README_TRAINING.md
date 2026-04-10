# SoulSync Training and Emotion Pipeline

This document covers every training-related component in the repository: embedding extraction, synthetic dataset creation, bridge model training, inference, file outputs, and limitations.

## 1) Pipeline Overview

SoulSync voice-emotion analysis uses two phases.

1. Feature extraction phase.
   - Script: train_emotion.py.
   - Model: Emotion2Vec loaded locally from models/emotion2vec.
   - Output: labels plus a dense embedding vector.

2. Classification bridge phase.
   - Script: train_bridge.py.
   - Model: TherapistBridge (PyTorch).
   - Output: 5-class emotion prediction from embedding.

Inference wrapper.
- Script: therapist_bot.py.
- Loads both phases and returns a human-readable emotion label.

Support scripts.
- generate_synthetic_data.py for training data creation.
- prep_data.py for metadata CSV scaffold.
- transcribe.py for STT sanity checking with Whisper.

## 2) Files and Roles

### train_emotion.py

- Defines EmotionProcessor.
- Device selection: mps if available, else cpu.
- Loads AutoModel(model="models/emotion2vec", disable_update=True).
- get_results(audio_path):
  - returns (None, None) if path missing.
  - runs model.generate(input=audio_path, extract_embedding=True).
  - returns labels dict and embedding array.
- Script mode behavior:
  - reads data/raw_audio/test.wav.
  - prints top emotion, confidence, embedding shape/preview.
  - saves embedding to data/test_embedding.npy.

### generate_synthetic_data.py

- Imports EmotionProcessor from train_emotion.py.
- Current anchor map:
  - data/raw_audio/test.wav -> label 4.
  - data/raw_audio/sad.wav -> label 1.
- For each anchor:
  - extracts base embedding.
  - generates 100 variants with Gaussian noise N(0, 0.02).
  - appends emb + noise to X and class label to y.
- Saves:
  - data/X_train.npy
  - data/y_train.npy
- Logs skip message if anchor file is missing.

### train_bridge.py

- Defines TherapistBridge network:
  - Linear(1024, 256)
  - ReLU
  - Dropout(0.2)
  - Linear(256, 5)
  - LogSoftmax(dim=1)
- train_bridge function:
  - checks data/X_train.npy exists (otherwise exits with message).
  - loads X and y from numpy.
  - optimizer: Adam(lr=0.001).
  - criterion: NLLLoss.
  - epochs: 50.
  - prints loss every 10 epochs.
  - saves checkpoint to models/checkpoints/bridge_v1.pth.

### therapist_bot.py

- Emotion map:
  - 0 Anxious/Stressed
  - 1 Sad/Depressed
  - 2 Angry/Frustrated
  - 3 Neutral/Calm
  - 4 Happy/Stable
- Loads:
  - EmotionProcessor encoder.
  - TherapistBridge with checkpoint models/checkpoints/bridge_v1.pth.
- analyze_voice(audio_path):
  - gets embedding from encoder.
  - converts to torch tensor with batch dimension.
  - runs forward pass and argmax.
  - maps class id to emotion string.
- Script mode:
  - tests data/raw_audio/sad.wav and prints interpretation.

### prep_data.py

- Writes data/metadata.csv with columns:
  - file_path
  - label
- Produces empty scaffold only (not wired into training loop yet).

### transcribe.py

- Loads Whisper base model on mps/cpu.
- transcribe(audio_path) returns stripped transcript.
- Script mode tests data/raw_audio/sad.wav.

## 3) Class Labels and Mapping

Bridge training/inference uses 5 classes.

- 0: Anxious/Stressed
- 1: Sad/Depressed
- 2: Angry/Frustrated
- 3: Neutral/Calm
- 4: Happy/Stable

Current synthetic generator only populates labels 1 and 4 by default.

## 4) Required Inputs and Artifacts

### Must exist before training/inference

- models/emotion2vec/config.yaml
- models/emotion2vec/configuration.json
- models/emotion2vec/model.pt
- models/emotion2vec/tokens.txt
- data/raw_audio/test.wav (for default scripts)
- data/raw_audio/sad.wav (for default scripts)

### Produced by scripts

- train_emotion.py -> data/test_embedding.npy
- generate_synthetic_data.py -> data/X_train.npy, data/y_train.npy
- train_bridge.py -> models/checkpoints/bridge_v1.pth

## 5) End-to-End Training Workflow

1. Optional: create metadata scaffold.

   python prep_data.py

2. Build synthetic dataset.

   python generate_synthetic_data.py

3. Train bridge classifier.

   python train_bridge.py

4. Run embedding sanity check.

   python train_emotion.py

5. Run inference sanity check.

   python therapist_bot.py

6. Optional STT audio quality check.

   python transcribe.py

## 6) Shapes, Devices, and Training Details

- Embedding dimension expected by bridge: 1024.
- Inference tensor shape before model call: (1, 1024).
- Device strategy in scripts: mps if torch.backends.mps.is_available() else cpu.
- Training is full-batch on the loaded arrays (no DataLoader).
- No train/validation split in current bridge training.
- Loss function assumes integer class labels in y_train.npy.

## 7) Integration Boundary with Main App

- Main web app currently uses text keyword emotion detection in brain.py for TTS conditioning.
- Voice-emotion classifier from therapist_bot.py is present but not yet wired into app.py request flow.
- This means live chat emotion label and trained voice-emotion prediction are separate paths today.

## 8) Dependency Notes (Training-Relevant)

- torch, torchvision, torchaudio: model training and inference.
- funasr, modelscope: Emotion2Vec loading and feature extraction.
- numpy: array storage for X/y and embeddings.
- librosa, soundfile: audio stack dependencies.
- whisper: transcription sanity checks.

## 9) Common Failure Modes

- Missing models/emotion2vec files -> AutoModel load fails.
- Missing anchor wav files -> synthetic generation skips or yields tiny dataset.
- Missing X_train.npy -> train_bridge.py exits with explicit error.
- Missing bridge_v1.pth -> therapist_bot.py load_state_dict fails.
- Corrupt/empty audio -> poor embedding quality and unstable predictions.

## 10) Practical Improvements

- Add anchors for all five classes (0-4).
- Replace synthetic-only training with real labeled multi-speaker recordings.
- Introduce train/validation split and evaluation metrics.
- Add class balancing and robust augmentations (time stretch, pitch shift, noise profiles).
- Route therapist_bot inference into app.py to drive TTS from voice emotion, not only text keywords.
- Add reproducibility controls (seed setting, config file, checkpoint versioning).
