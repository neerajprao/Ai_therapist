# Training and evaluating the emotion model

This describes the pipeline that produced the numbers in [RESULTS.md](RESULTS.md). Every step is a script; run them from the repo root with the virtual environment active.

```bash
python3.11 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements-train.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1     # Apple silicon
```

## 1. Get the data

The audio is not in this repository. Download the Kaggle dataset `uldisvalainis/audio-emotions` and place its `Emotions/` folder at `custom_model/Emotions/` (git-ignored).

```python
import kagglehub
print(kagglehub.dataset_download("uldisvalainis/audio-emotions"))
```

## 2. Inventory

```bash
python -m training.inventory
```

Parses every filename into corpus, speaker and emotion, hashes the audio, and writes `training/splits/dataset_index.csv`, `eval/results/dataset_inventory.json` and [docs/DATASET.md](docs/DATASET.md).

The emotion label comes from the filename, not the folder. In this Kaggle packaging the SAVEE clips are filed under the wrong class folders, so folder labels are wrong for most of them.

## 3. Speaker-independent split

```bash
python -m training.make_splits
```

Writes `training/splits/split_v2.csv` and `eval/results/splits.json`.

- Five labels: happy, sad, angry, neutral, anxious (the corpora's "fearful"). Disgusted and surprised clips are dropped.
- RAVDESS song clips are excluded; the app hears speech.
- Identical audio filed under two different labels is dropped.
- No speaker appears in more than one of train, validation and test. Seed 42.
- TESS has two speakers with 1,000 clips each, so both stay in train.
- It also writes `split_v1_repro.csv`, which rebuilds the v1 notebook's data recipe (RAVDESS only, split by file) while keeping the same test set.

## 4. Cache waveforms

```bash
python -m training.data
```

Decodes every five-class clip to 16 kHz mono once and stores it in `training/cache/` (about 850 MB). Epochs then do no audio decoding. The frozen CNN's outputs are not cached, because training uses a random 3-second crop of the waveform each epoch.

## 5. Train

```bash
python -m training.train --run-name main --bench-steps 60      # timing benchmark, saves nothing
caffeinate -i python -m training.train --run-name main --epochs 15 --patience 3
python -m training.train --run-name main --epochs 15 --resume  # continue after an interruption
```

| Setting | Value |
|---|---|
| Base model | `facebook/wav2vec2-base`, CNN feature encoder frozen |
| Input | 3 s at 16 kHz, zero-mean unit-variance per clip, zero-padded; random crop for longer clips |
| Loss | cross-entropy with inverse-frequency class weights |
| Optimiser | AdamW, lr 5e-5, weight decay 0.01, 10% linear warm-up then linear decay, gradient clipping 1.0 |
| Batch | 16, fp32 on MPS |
| Model selection | best validation macro-F1, early stopping after 3 epochs without improvement |
| Checkpoints | `training/runs/<run>/last` every epoch (resumable) and `.../best` |

Per-epoch metrics and wall-clock time go to `eval/results/training_<run>.json`. The training script never reads the test split.

Preprocessing lives in one place, `soulsync/emotion.py` (`normalize_clip`, `make_windows`), and is used by training, evaluation, the PyTorch backend and the ONNX backend.

### Reproducing the v1 recipe as a baseline

```bash
python -m training.train --run-name v1_repro --split split_v1_repro.csv \
  --base-model facebook/wav2vec2-base-960h --epochs 10 --batch-size 4 --accum 4 \
  --lr 3e-5 --warmup-frac 0 --patience 10 --select accuracy --no-random-crop --no-class-weights
```

The original v1 weights were not available, so this is a reproduction of the recipe, not the original model.

### Cross-corpus run

```bash
python -m training.train --run-name no_tess --exclude-corpus TESS --epochs 8 --patience 3
python -m eval.emotion_eval --model-dir training/runs/no_tess/best --only-corpus TESS --name cross_corpus_tess
```

## 6. Evaluate

```bash
python -m eval.emotion_eval --model-dir training/runs/v1_repro/best --split-csv split_v1_repro.csv --name baseline_v1_repro
python -m eval.emotion_eval --model-dir training/runs/main/best --name test
python -m eval.ensemble_eval --model-dir training/runs/main/best    # needs models/emotion2vec/model.pt (Git LFS)
```

Evaluation runs the same windowing as the app: one padded window for clips up to 3 s, otherwise overlapping 3 s windows whose probabilities are averaged. Each script writes JSON, a short log and (for `emotion_eval`) a confusion-matrix PNG to `eval/results/`.

The test split is for the final report only. Choices such as epoch count and ensemble weight are made on validation.

## 7. Export to ONNX and quantise

```bash
cp -R training/runs/main/best models/emotion_w2v2
python -m training.export_onnx
python -m eval.onnx_benchmark
```

Produces `models/emotion_w2v2_fp32.onnx` and `models/emotion_w2v2_int8.onnx`. `EMOTION_BACKEND=onnx` then runs without PyTorch.

## 8. Regenerate the results page

```bash
python -m eval.memory_eval
python -m eval.latency_eval --compare-stt
python -m eval.make_results
```

`RESULTS.md` is built only from the JSON files in `eval/results/`.
