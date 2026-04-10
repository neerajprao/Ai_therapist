# SoulSync Training & Emotion Pipeline Guide

This README is focused exclusively on the training, data preparation, and emotion analysis workflow for the SoulSync project.

## Purpose

SoulSync uses a two-phase voice emotion pipeline:

1. **Phase 1: Emotion Embedding Extraction**
   - Uses `models/emotion2vec` and `train_emotion.py`
   - Converts audio into an emotion-aware embedding vector (1024D)

2. **Phase 2: Emotion Bridge Network**
   - Uses `train_bridge.py`
   - Trains a small classifier to map embeddings to one of 5 emotion categories

These two phases power `therapist_bot.py`, which can classify recorded voice into emotional labels. The pipeline is separate from the text-based emotion detection in `brain.py` used for TTS.

## File Summary

- `train_emotion.py`
  - Loads `models/emotion2vec` via FunASR AutoModel on MPS/CPU
  - Uses `self.model.generate(input=audio_path, extract_embedding=True)` to get labels and embeddings
  - Saves `data/test_embedding.npy` during test mode
  - Device detection: MPS for Apple Silicon, else CPU

- `train_bridge.py`
  - Defines the `TherapistBridge` neural network: Linear(1024, 256) → ReLU → Dropout(0.2) → Linear(256, 5) → LogSoftmax
  - Trains on `data/X_train.npy` / `data/y_train.npy` with Adam (lr=0.001), NLLLoss, 50 epochs
  - Saves weights to `models/checkpoints/bridge_v1.pth`
  - Prints loss every 10 epochs

- `generate_synthetic_data.py`
  - Creates synthetic dataset using anchor audio files
  - Adds Gaussian noise (μ=0, σ=0.02) to each embedding for 100 variations per class
  - Produces `data/X_train.npy` (float32 embeddings) and `data/y_train.npy` (int64 labels)
  - Currently uses only 2 classes: Sad (1), Happy (4)

- `prep_data.py`
  - Generates empty `data/metadata.csv` with `file_path` and `label` columns
  - Intended for manual labeling but not integrated into training

- `therapist_bot.py`
  - Loads Emotion2Vec encoder and bridge weights
  - Exposes `analyze_voice(audio_path)` returning human-readable emotion string
  - Maps class indices to EMOTION_MAP: 0→Anxious, 1→Sad, 2→Angry, 3→Neutral, 4→Happy

- `transcribe.py`
  - Loads Whisper base model on MPS/CPU
  - Transcribes audio to text for testing
  - Useful for verifying audio quality before emotion training

## Emotion Classes

The bridge predicts one of five classes (matching `therapist_bot.py`):

- `0`: Anxious/Stressed
- `1`: Sad/Depressed
- `2`: Angry/Frustrated
- `3`: Neutral/Calm
- `4`: Happy/Stable

These are defined in `EMOTION_MAP` for inference.

## Setup Requirements

Before running any training or inference:

- Install packages from `requirements.txt` (torch, funasr, librosa, numpy, pandas)
- Ensure `models/emotion2vec/` exists with model files (config.yaml, model.pt, tokens.txt)
- Ensure `data/raw_audio/` contains anchor audio files (e.g., test.wav, sad.wav)
- Python 3.8+, torch with MPS support for Mac

Optional but recommended:
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

## Workflow: From Audio to Emotion Prediction

### 1. Prepare raw audio

Place or record labeled audio files in `data/raw_audio/`.

Example anchor files used by current scripts:

- `data/raw_audio/test.wav` → Happy/Stable (label 4)
- `data/raw_audio/sad.wav` → Sad/Depressed (label 1)

Audio should be WAV format, clear voice samples.

### 2. Create metadata placeholder

```bash
python prep_data.py
```

Creates `data/metadata.csv` with empty rows for `file_path` and `label`. Not currently used in pipeline.

### 3. Extract embeddings and create synthetic dataset

```bash
python generate_synthetic_data.py
```

What this does:

- Initializes `EmotionProcessor` with Emotion2Vec model
- For each anchor file: extracts embedding, generates 100 noisy copies (noise ~ N(0, 0.02))
- Saves to NumPy arrays: X (shape: [200, 1024]), y (shape: [200,])

Important notes:

- Script skips missing files with warning
- Noise level (0.02) creates variation without destroying structure
- Extend by adding more targets to the `targets` dict with labels 0-4

### 4. Train the bridge classifier

```bash
python train_bridge.py
```

Training details:

- Network: 1024→256→5 with ReLU, Dropout(0.2), LogSoftmax
- Loss: Negative Log Likelihood (NLLLoss)
- Optimizer: Adam, lr=0.001
- Epochs: 50 (prints loss every 10)
- Device: MPS if available, else CPU

Output:

- `models/checkpoints/bridge_v1.pth` (state_dict)

Aborts if `data/X_train.npy` not found.

### 5. Verify using therapist_bot.py

```bash
python therapist_bot.py
```

Behavior:

- Loads encoder and bridge on device
- Analyzes `data/raw_audio/sad.wav`
- Prints predicted emotion (e.g., "Sad/Depressed")

This is end-to-end test of the pipeline.

## `train_emotion.py` Detailed Behavior

The `EmotionProcessor` class:

- Loads model from `models/emotion2vec` with `AutoModel`
- Device: MPS for Apple Silicon, CPU otherwise
- `get_results(audio_path)`: calls `model.generate(..., extract_embedding=True)`
- Returns: labels dict (e.g., {'happy': 0.85, ...}), embeddings np.array (1024,)

When executed directly:

- Processes `data/raw_audio/test.wav`
- Prints top emotion and confidence
- Prints embedding shape and preview
- Saves embedding to `data/test_embedding.npy`

## `train_bridge.py` Architecture Notes

The bridge converts Emotion2Vec embedding to emotion class.

Model definition (in `TherapistBridge`):

- `nn.Linear(1024, 256)`
- `nn.ReLU()`
- `nn.Dropout(0.2)`
- `nn.Linear(256, num_classes)`  # 5
- `nn.LogSoftmax(dim=1)`

Training loop:

- Loads X, y as torch tensors on device
- Fixed 50 epochs with zero_grad, forward, loss, backward, step
- No validation split; simple overfitting to synthetic data

## Inference with `therapist_bot.py`

`TherapistBot.analyze_voice(audio_path)`:

1. `EmotionProcessor.get_results(audio_path)` → labels, embeddings
2. Convert embeddings to tensor (1, 1024) on device
3. Forward through `TherapistBridge`
4. Argmax to get class index
5. Map to string via `EMOTION_MAP`

This is the voice emotion prediction used separately from text emotion.

## Extending the Pipeline

Recommended improvements:

- Add more anchor files for all 5 emotions
- Replace synthetic noise with real labeled audio
- Implement `data/metadata.csv` labeling workflow
- Add validation split and early stopping in training
- Experiment with larger networks or pre-trained classifiers
- Add data augmentation (pitch shift, speed change)
- Integrate voice emotion into main app (currently text-based for TTS)

## Notes and Caveats

- Pipeline is prototype with synthetic data from 2 classes
- Emotion2Vec model must be in `models/emotion2vec/` (not included; download separately)
- `generate_synthetic_data.py` skips missing anchors silently
- `train_bridge.py` requires synthetic data first
- `prep_data.py` creates CSV but doesn't populate it
- Tested on Mac with MPS; adjust device for Windows/Linux

## Quick Command Reference

- `python prep_data.py` — Create metadata CSV
- `python generate_synthetic_data.py` — Generate synthetic X/y
- `python train_bridge.py` — Train bridge network
- `python train_emotion.py` — Test embedding extraction
- `python therapist_bot.py` — Test emotion prediction
- `python transcribe.py` — Test Whisper transcription

## Recommended File Additions

For stronger training:

- More `data/raw_audio/` samples across all 5 emotions
- Populated `data/metadata.csv` with real labels
- `README_DATA.md` for dataset documentation
- Validation/test splits in training
- Model evaluation metrics (accuracy, F1)

## Extending the Pipeline

Recommended improvements:

- Add more anchor files for all 5 emotions
- Replace synthetic noise with real labeled audio
- Implement `data/metadata.csv` labeling workflow
- Add validation split and early stopping in training
- Experiment with larger networks or pre-trained classifiers
- Add data augmentation (pitch shift, speed change)
- Integrate voice emotion into main app (currently text-based for TTS)

## Notes and Caveats

- Pipeline is prototype with synthetic data from 2 classes
- Emotion2Vec model must be in `models/emotion2vec/` (not included; download separately)
- `generate_synthetic_data.py` skips missing anchors silently
- `train_bridge.py` requires synthetic data first
- `prep_data.py` creates CSV but doesn't populate it
- Tested on Mac with MPS; adjust device for Windows/Linux

## Quick Command Reference

- `python prep_data.py` — Create metadata CSV
- `python generate_synthetic_data.py` — Generate synthetic X/y
- `python train_bridge.py` — Train bridge network
- `python train_emotion.py` — Test embedding extraction
- `python therapist_bot.py` — Test emotion prediction
- `python transcribe.py` — Test Whisper transcription

## Recommended File Additions

For stronger training:

- More `data/raw_audio/` samples across all 5 emotions
- Populated `data/metadata.csv` with real labels
- `README_DATA.md` for dataset documentation
- Validation/test splits in training
- Model evaluation metrics (accuracy, F1)
