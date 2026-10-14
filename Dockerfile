# Cloud-mode image: no PyTorch. Groq for LLM and speech-to-text, ONNX INT8
# emotion model, ChromaDB memory, browser speech output.
# Size and memory are measured by `python -m eval.cloud_memory`.
# The emotion model is downloaded from the Hugging Face Hub at build time (pinned
# revision, SHA-256 checked), so the image builds straight from the git repo.
#   docker build -t soulsync-cloud .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODE=cloud \
    DATA_DIR=/data \
    OMP_NUM_THREADS=1

RUN useradd --create-home app && mkdir -p /data /app && chown app /data /app
WORKDIR /app

COPY requirements-cloud.txt .
RUN pip install -r requirements-cloud.txt

USER app
# Fetch the MiniLM embedding model at build time, into the app user's cache,
# so containers start without a download.
RUN python -c "from chromadb.utils import embedding_functions as e; e.DefaultEmbeddingFunction()(['warm up'])"

ARG MODEL_REPO=npr1138/soulsync-emotion-wav2vec2
ARG MODEL_REVISION=81ceb25603272e723c90eb06d6ef15a987688005
ARG MODEL_SHA256=e4552a09e40c6907734e930b7b749eeab7967f945660a52f0356a5da58aa260a
RUN python - <<PY
import hashlib, os, urllib.request
url = "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/emotion_w2v2_int8.onnx"
os.makedirs("models", exist_ok=True)
urllib.request.urlretrieve(url, "models/emotion_w2v2_int8.onnx")
digest = hashlib.sha256(open("models/emotion_w2v2_int8.onnx", "rb").read()).hexdigest()
assert digest == "${MODEL_SHA256}", f"checksum mismatch: {digest}"
print("emotion model downloaded and verified")
PY
COPY --chown=app soulsync ./soulsync

EXPOSE 8000
CMD ["sh", "-c", "uvicorn soulsync.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
