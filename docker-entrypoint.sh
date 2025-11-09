#!/usr/bin/env bash
set -euo pipefail

# create persistent dirs if they don't exist (and set permissions)
mkdir -p /data /cache
chown -R "$(id -u):$(id -g)" /data /cache || true

# allow overriding the CMD with environment variables
# e.g., RUN_MODE=gradio to launch Gradio, default is "api"
RUN_MODE=${RUN_MODE:-api}
DEV_MODE=${DEV_MODE:-0}

if [ "$RUN_MODE" = "gradio" ]; then
  # launch gradio app (module path)
  # make sure ui.gradio_app exposes a proper launch when run as module
  if [ "$DEV_MODE" = "1" ]; then
    exec python -m ui.gradio_app
  else
    # non-dev: run gradio in headless mode
    exec python -m ui.gradio_app
  fi
else
  # default: run uvicorn for FastAPI
  if [ "$DEV_MODE" = "1" ]; then
    exec uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
  else
    exec uvicorn src.app.main:app --host 127.0.0.1 --port 8000
  fi
fi
