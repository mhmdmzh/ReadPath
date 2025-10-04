# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Minimal OS deps (libgomp helps scipy/sklearn wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Preinstall CPU-only PyTorch (keep torch OUT of requirements.txt)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# App dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Optional warm-up (SBERT + NLTK). Disabled by default.
# Enable with: docker build --build-arg WARMUP=1 -t readpath-app .
ARG WARMUP=0
RUN if [ "$WARMUP" = "1" ]; then \
      python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); \
from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Warm-up done')"; \
    fi

# App code
COPY app.py .

EXPOSE 7860
CMD ["python", "app.py"]
