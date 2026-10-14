# Cloud-mode image: no PyTorch. Groq for LLM and speech-to-text, ONNX INT8
# emotion model, ChromaDB memory, browser speech output.
#
# NOT YET BUILT OR MEASURED: Docker was not available when this was written.
# Build from the repo root after exporting the ONNX model:
#   python -m training.export_onnx
#   docker build -t soulsync-cloud .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODE=cloud \
    DATA_DIR=/data \
    OMP_NUM_THREADS=1

WORKDIR /app
COPY requirements-cloud.txt .
RUN pip install -r requirements-cloud.txt

# Fetch the MiniLM embedding model at build time so the first request is not a download.
RUN python -c "from chromadb.utils import embedding_functions as e; e.DefaultEmbeddingFunction()(['warm up'])"

COPY soulsync ./soulsync
COPY models/emotion_w2v2_int8.onnx ./models/emotion_w2v2_int8.onnx

RUN useradd --create-home app && mkdir -p /data && chown -R app /data /app
USER app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn soulsync.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
