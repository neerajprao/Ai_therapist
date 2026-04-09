# SoulSync Training & Emotion Pipeline Guide

This README is focused exclusively on the training, data preparation, and emotion analysis workflow for the SoulSync project.

## Purpose

SoulSync uses a two-phase voice emotion pipeline:

1. **Phase 1: Emotion Embedding Extraction**
   - Uses `models/emotion2vec` and `train_emotion.py`
   - Converts audio into an emotion-aware embedding vector

2. **Phase 2: Emotion Bridge Network**
   - Uses `train_bridge.py`
   - Trains a small classifier to map embeddings to one of 5 emotion categories

These two phases power `therapist_bot.py`, which can classify recorded voice into emotional labels.

## File Summary

- `train_emotion.py`
  - Loads `models/emotion2vec`
  - Uses `funasr.AutoModel` to generate emotion labels and embeddings from audio
  - Saves `data/test_embedding.npy` during test mode

- `train_bridge.py`
  - Defines the `TherapistBridge` neural network
  - Trains on `data/X_train.npy` / `data/y_train.npy`
  - Saves weights to `models/checkpoints/bridge_v1.pth`

- `generate_synthetic_data.py`
  - Creates a synthetic dataset using a few anchor audio files
  - Adds random Gaussian noise to each embedding to build a larger dataset
  - Produces `data/X_train.npy` and `data/y_train.npy`

- `prep_data.py`
  - Generates a skeleton metadata CSV at `data/metadata.csv`
  - Intended as a starting point for labeling raw audio files

- `therapist_bot.py`
  - Loads the Emotion2Vec encoder and trained bridge
  - Exposes `analyze_voice(audio_path)` returning a human-readable emotion label

- `transcribe.py`
  - Local Whisper transcription helper
  - Useful for verifying audio quality and debug transcriptions

## Emotion Classes

The bridge is trained to predict one of five classes:

- `0`: Anxious/Stressed
- `1`: Sad/Depressed
- `2`: Angry/Frustrated
- `3`: Neutral/Calm
- `4`: Happy/Stable

These classes are defined in `therapist_bot.py` as `EMOTION_MAP`.

## Setup Requirements

Before running any training or inference:

- Install packages from `requirements.txt`
- Ensure `models/emotion2vec/` exists and is populated with the Emotion2Vec model files
- Ensure `data/raw_audio/` contains anchor audio files such as `test.wav` and `sad.wav`

Optional but recommended:
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

## Workflow: From Audio to Emotion Prediction

### 1. Prepare raw audio

Place or record labeled audio files in `data/raw_audio/`.

Example anchor files used by the current scripts:

- `data/raw_audio/test.wav` → Happy/Stable anchor
- `data/raw_audio/sad.wav` → Sad/Depressed anchor

### 2. Create metadata placeholder

```bash
python prep_data.py
```

This writes a fresh `data/metadata.csv` with columns:

- `file_path`
- `label`

Use this CSV to track or expand your dataset labeling process.

### 3. Extract embeddings and create a synthetic dataset

```bash
python generate_synthetic_data.py
```

What this does:

- Loads the Emotion2Vec model via `EmotionProcessor`
- Extracts an embedding for each anchor audio file
- Generates 100 noisy variations of each embedding
- Saves datasets to `data/X_train.npy` and `data/y_train.npy`

Important notes:

- The script currently uses only two labels: `1` for sad and `4` for happy.
- It applies Gaussian noise to simulate data diversity.
- You can extend this script to use more anchor files and more emotion classes.

### 4. Train the bridge classifier

```bash
python train_bridge.py
```

Training details:

- Network architecture: `1024 -> 256 -> 5`
- Activation: ReLU
- Output: LogSoftmax
- Loss: Negative log likelihood (`NLLLoss`)
- Optimizer: Adam, learning rate `0.001`
- Epochs: 50

Output:

- `models/checkpoints/bridge_v1.pth`

If the data file does not exist, the script aborts with an error message.

### 5. Verify using therapist_bot.py

```bash
python therapist_bot.py
```

Behavior:

- Loads the Emotion2Vec encoder and bridge weights
- Attempts to analyze `data/raw_audio/sad.wav`
- Prints a predicted emotion label

This script is a quick end-to-end sanity check for the emotion pipeline.

## `train_emotion.py` Detailed Behavior

The `EmotionProcessor` class:

- Detects available device: `mps` if on Apple silicon, otherwise `cpu`
- Loads a local Emotion2Vec model from `models/emotion2vec`
- Runs `self.model.generate(input=audio_path, extract_embedding=True)`
- Returns a label score dictionary and a NumPy embedding vector

When executed directly, it will:

- Load `data/raw_audio/test.wav`
- Print a report of top emotion labels and embedding shape
- Save the embedding into `data/test_embedding.npy`

## `train_bridge.py` Architecture Notes

The bridge exists to convert the Emotion2Vec embedding into a discrete emotion class.

Model definition:

- `Linear(1024, 256)`
- `ReLU()`
- `Dropout(0.2)`
- `Linear(256, num_classes)`
- `LogSoftmax(dim=1)`

Training loop:

- Loads `data/X_train.npy` and `data/y_train.npy`
- Converts them to Torch tensors
- Runs a fixed 50-epoch training cycle
- Prints loss every 10 epochs
- Saves final weights to `models/checkpoints/bridge_v1.pth`

## Inference with `therapist_bot.py`

`TherapistBot.analyze_voice(audio_path)` does:

1. `EmotionProcessor.get_results(audio_path)` to extract voice embeddings
2. Converts embeddings to a `(1, 1024)` tensor
3. Feeds the tensor to `TherapistBridge`
4. Selects the class with highest score
5. Maps the class index to a human-readable label using `EMOTION_MAP`

This is the inference path for voice emotion classification.

## Extending the Pipeline

Recommended improvements:

- Add more anchor files to `generate_synthetic_data.py`
- Replace synthetic augmentation with real labeled audio
- Add a true `data/metadata.csv` labeling workflow
- Increase training epochs or add validation split
- Use a better classifier than a simple single-hidden-layer network
- Add logging and error handling for missing files and invalid audio

## Notes and Caveats

- The current pipeline is mostly prototype-level and uses limited synthetic data.
- The Emotion2Vec model must be present in `models/emotion2vec/` for `train_emotion.py` and `therapist_bot.py` to work.
- `generate_synthetic_data.py` will silently skip missing anchor audio files.
- `train_bridge.py` assumes the synthetic dataset already exists.
- `prep_data.py` does not actually label audio; it only creates an empty CSV.

## Quick Command Reference

- `python prep_data.py`
- `python generate_synthetic_data.py`
- `python train_bridge.py`
- `python train_emotion.py`
- `python therapist_bot.py`
- `python transcribe.py`

## Recommended File Additions

For a stronger training workflow, add:

- `data/metadata.csv` with actual labeled audio rows
- more `data/raw_audio/` samples across all five emotion classes
- `README_DATA.md` or annotated dataset documentation if you scale beyond prototypes
