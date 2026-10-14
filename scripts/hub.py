"""Upload the trained emotion model to the Hugging Face Hub, or download it.

  hf auth login                                  # once, with a write token
  python scripts/hub.py upload                   # -> <you>/soulsync-emotion-wav2vec2
  python scripts/hub.py download --repo <user>/soulsync-emotion-wav2vec2

Upload sends: the PyTorch model (config + safetensors), the INT8 ONNX model,
the confusion-matrix image and a model card built from docs/MODEL_CARD.md.
Download puts the files where the app expects them under models/.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_NAME = "soulsync-emotion-wav2vec2"

FRONT_MATTER = """---
license: cc-by-nc-sa-4.0
language: en
library_name: transformers
pipeline_tag: audio-classification
base_model: facebook/wav2vec2-base
tags:
- speech-emotion-recognition
- wav2vec2
- onnx
- int8
metrics:
- accuracy
- f1
---

"""


def build_card(github_url: str) -> str:
    card = (ROOT / "docs" / "MODEL_CARD.md").read_text()
    card = card.replace("../eval/results/emotion_test.png", "emotion_test.png")
    card = card.replace("(DATASET.md)", f"({github_url}/blob/v2/docs/DATASET.md)")
    usage = f"""
## Files

| File | Use |
|---|---|
| `config.json`, `model.safetensors` | PyTorch / transformers (`Wav2Vec2ForSequenceClassification`) |
| `emotion_w2v2_int8.onnx` | onnxruntime, CPU, no PyTorch needed |

Input must be preprocessed exactly as in the app: 16 kHz mono, 3 s windows, zero-mean unit-variance per window, zero-padded (see `soulsync/emotion.py` in [the code repository]({github_url})).

```bash
python scripts/hub.py download --repo <this repo id>
```

"""
    return FRONT_MATTER + card.replace("## Data\n", usage + "## Data\n", 1)


def upload(private: bool, github_url: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{REPO_NAME}"
    files = {
        ROOT / "models" / "emotion_w2v2" / "config.json": "config.json",
        ROOT / "models" / "emotion_w2v2" / "model.safetensors": "model.safetensors",
        ROOT / "models" / "emotion_w2v2_int8.onnx": "emotion_w2v2_int8.onnx",
        ROOT / "eval" / "results" / "emotion_test.png": "emotion_test.png",
    }
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for src, name in files.items():
            shutil.copy(src, tmp / name)
        (tmp / "README.md").write_text(build_card(github_url))
        api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="model",
                          commit_message="Upload SoulSync emotion model (v2)")
    print(f"uploaded: https://huggingface.co/{repo_id}")


def download(repo_id: str) -> None:
    from huggingface_hub import hf_hub_download

    target = ROOT / "models" / "emotion_w2v2"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors"):
        shutil.copy(hf_hub_download(repo_id, name), target / name)
    shutil.copy(hf_hub_download(repo_id, "emotion_w2v2_int8.onnx"), ROOT / "models" / "emotion_w2v2_int8.onnx")
    print(f"downloaded {repo_id} into models/")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--private", action="store_true")
    up.add_argument("--github-url", default="https://github.com/neerajprao/Ai_therapist")
    down = sub.add_parser("download")
    down.add_argument("--repo", required=True)
    args = p.parse_args()
    if args.cmd == "upload":
        upload(args.private, args.github_url)
    else:
        download(args.repo)


if __name__ == "__main__":
    main()
