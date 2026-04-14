# SoulSync Training Pipeline

This document describes the offline Emotion2Vec plus bridge pipeline exactly as implemented in this repository. It covers the feature extractor, the synthetic dataset generator, the bridge classifier, the offline inference wrapper, and the helper scripts that support them.

## 1. What The Training Stack Is For

The training stack maps raw voice audio to one of five therapist-facing categories.

The pipeline is intentionally split into two stages:

1. Stage A: `EmotionProcessor` extracts a dense embedding from local Emotion2Vec.
2. Stage B: `TherapistBridge` maps that embedding to a five-class emotion label.

The offline wrapper `therapist_bot.py` loads both stages and returns a readable label for a test audio file.

## 2. Files In This Stack

- `train_emotion.py`
  - Emotion2Vec wrapper and embedding extractor.
- `generate_synthetic_data.py`
  - Synthetic data generator for the bridge classifier.
- `train_bridge.py`
  - Bridge classifier definition and training loop.
- `therapist_bot.py`
  - Offline inference wrapper.
- `transcribe.py`
  - Standalone Whisper transcription test.
- `prep_data.py`
  - CSV scaffold creator.

## 3. `train_emotion.py`

Purpose:

- Load local Emotion2Vec with FunASR and return both label scores and raw embeddings.

Implementation details:

- Device selection:
  - `mps` if available
  - otherwise `cpu`
- Local model path:
  - `models/emotion2vec`
- Model initialization:
  - `AutoModel(model=model_path, device=device, disable_update=True)`
- `get_results(audio_path)`:
  - returns `(None, None)` if the file is missing
  - calls `self.model.generate(input=audio_path, extract_embedding=True)`
  - converts the returned label list and score list into a dictionary
  - converts the embedding tensor-like output into a NumPy array

Script mode (`python train_emotion.py`):

1. Uses `data/raw_audio/test.wav`.
2. Prints the strongest Emotion2Vec label and its confidence.
3. Prints the embedding shape.
4. Prints a short preview of the feature vector.
5. Saves the embedding to `data/test_embedding.npy`.

Important note:

- The module imports `librosa`, but the current code path does not use it.

## 4. `generate_synthetic_data.py`

Purpose:

- Build a small synthetic training set for the bridge classifier from a handful of anchor audio files.

Hardcoded anchors:

- `data/raw_audio/test.wav -> 4`
- `data/raw_audio/sad.wav -> 1`

Behavior:

1. Instantiates `EmotionProcessor`.
2. For each anchor that exists, extracts the base embedding.
3. Creates 100 noisy variants of that embedding with Gaussian noise `N(0, 0.02)`.
4. Appends the noisy samples to `X` and the class labels to `y`.
5. Saves the arrays to:
   - `data/X_train.npy`
   - `data/y_train.npy`

Important implication:

- The generated dataset only contains two classes by default, even though the bridge outputs five classes.

## 5. `train_bridge.py`

Purpose:

- Train the small feed-forward classifier that maps Emotion2Vec embeddings to five emotion classes.

Model definition:

- `TherapistBridge(input_dim=1024, num_classes=5)`
- Layers:
  - `Linear(1024, 256)`
  - `ReLU`
  - `Dropout(0.2)`
  - `Linear(256, 5)`
  - `LogSoftmax(dim=1)`

Training loop:

- Loads `data/X_train.npy` and `data/y_train.npy`.
- Uses `mps` if available, otherwise `cpu`.
- Optimizer: `Adam(lr=0.001)`.
- Loss: `NLLLoss`.
- Epochs: `50`.
- Logs every 10 epochs.
- Saves the checkpoint to `models/checkpoints/bridge_v1.pth`.

Failure behavior:

- If `data/X_train.npy` is missing, the script prints an error message and exits early.

Implementation note:

- The model is trained full-batch, not with a `DataLoader`.

## 6. `therapist_bot.py`

Purpose:

- Run offline inference on a single audio file.

Inference flow:

1. Load `EmotionProcessor`.
2. Load `TherapistBridge` from `models/checkpoints/bridge_v1.pth`.
3. Get the Emotion2Vec embedding for the input file.
4. Reshape the embedding to `(1, 1024)`.
5. Run the bridge forward pass.
6. Take `argmax` of the output.
7. Convert the class ID into a human-readable state with `EMOTION_MAP`.

Class mapping:

- `0 -> Anxious/Stressed`
- `1 -> Sad/Depressed`
- `2 -> Angry/Frustrated`
- `3 -> Neutral/Calm`
- `4 -> Happy/Stable`

Script mode test file:

- `data/raw_audio/sad.wav`

Important difference from the live app:

- `therapist_bot.py` does not check for a missing checkpoint before loading it, so a missing `bridge_v1.pth` will raise an error.

## 7. `transcribe.py`

Purpose:

- Provide a standalone Whisper smoke test for the sample audio.

Behavior:

- Loads Whisper `base`.
- Picks `mps` if available, otherwise `cpu`.
- Moves the model to that device.
- Transcribes `data/raw_audio/sad.wav`.
- Prints the transcript.

## 8. `prep_data.py`

Purpose:

- Create an empty CSV scaffold for future audio metadata.

Output:

- `data/metadata.csv`

Columns:

- `file_path`
- `label`

Current status:

- No current training script consumes this CSV yet.

## 9. Mathematical Summary

For an embedding vector $x \in \mathbb{R}^{1024}$, the bridge computes logits for five classes with a single hidden layer and dropout:

$$
\hat{y} = \log\text{softmax}(W_2(\text{Dropout}(\text{ReLU}(W_1 x + b_1))) + b_2)
$$

The training objective is negative log likelihood:

$$
\mathcal{L} = -\log p(c_{true} \mid x)
$$

where $c_{true}$ is the integer label.

## 10. End-To-End Training Order

Use the scripts in this order if you want to reproduce the bridge pipeline:

```bash
source venv/bin/activate
python prep_data.py
python generate_synthetic_data.py
python train_bridge.py
python train_emotion.py
python therapist_bot.py
python transcribe.py
```

## 11. Required Files

### 11.1 Local Emotion2Vec Directory

Must contain:

- `models/emotion2vec/config.yaml`
- `models/emotion2vec/configuration.json`
- `models/emotion2vec/model.pt`
- `models/emotion2vec/tokens.txt`

### 11.2 Sample Audio Inputs Used By Default Scripts

- `data/raw_audio/test.wav`
- `data/raw_audio/sad.wav`

### 11.3 Inference Checkpoint

- `models/checkpoints/bridge_v1.pth`

## 12. Generated Files

The pipeline creates or updates these files:

- `data/test_embedding.npy`
- `data/X_train.npy`
- `data/y_train.npy`
- `models/checkpoints/bridge_v1.pth`
- `data/metadata.csv`

## 13. How This Relates To The Live App

The live runtime in `brain.py` uses the same basic ingredients:

- `EmotionProcessor` for embedding extraction
- `TherapistBridge` for fallback classification
- a hardcoded keyword map for text-first emotion detection

The key difference is that the web app checks keywords first and only falls back to the bridge if no keyword match is found.

## 14. Current Limitations

1. The synthetic dataset is tiny and only covers two labels by default.
2. There is no validation split.
3. There is no test set or reported accuracy.
4. Random augmentation is not seeded, so repeated runs can produce different training arrays.
5. The bridge is trained full-batch on a very small sample set.
6. The neural fallback in the live app catches all exceptions and silently returns `Neutral`.
7. The offline scripts assume the checkpoint already exists unless you train it first.
8. The pipeline depends on local model files under `models/emotion2vec/`.

## 15. Recommended Improvements

1. Add anchors for all five classes.
2. Add a deterministic seed for NumPy and PyTorch.
3. Add a train/validation split and log metrics.
4. Add richer audio augmentation.
5. Replace broad exception handling with explicit error reporting.
6. Document the expected audio format for the sample files more clearly.
