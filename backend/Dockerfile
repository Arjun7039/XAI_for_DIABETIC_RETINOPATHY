FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY weights/ ./weights/

EXPOSE 8000

ENV PORT=8000
ENV MODEL_CONFIG_PATH=weights/efficientnet_b4_config.json
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV MALLOC_ARENA_MAX=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 120
