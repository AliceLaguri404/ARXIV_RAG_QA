# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ---------- system deps ----------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential git curl wget libgl1 libglib2.0-0 cargo pkg-config libxml2-dev libxslt-dev libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------- create non-root user (root context) ----------
ARG APP_USER=appuser
ARG APP_UID=1000
RUN useradd --create-home --uid ${APP_UID} ${APP_USER}

WORKDIR /app

# ---------- copy requirements and install (root) ----------
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---------- copy source (root) ----------
COPY . /app

# ---------- create mountable dirs and set ownership (root) ----------
RUN mkdir -p /data /cache /tmp/cache \
 && chmod 775 /data /cache /tmp/cache \
 && chown -R ${APP_USER}:${APP_USER} /data /cache /tmp/cache /app

# ---------- copy entrypoint and set perms (still root) ----------
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh \
 && chown ${APP_USER}:${APP_USER} /app/docker-entrypoint.sh

# ---------- switch to unprivileged user ----------
USER ${APP_USER}

# ---------- env defaults ----------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_PERSIST_PATH=/data/vectorstore \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    GRADIO_SERVER_PORT=7860

# ---------- expose ports ----------
EXPOSE 8000 7860

# ---------- entrypoint ----------
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# default CMD runs FastAPI; override to run gradio or other entrypoints
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
