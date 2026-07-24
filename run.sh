#!/usr/bin/env bash
# Start the OpenAI -> Anthropic proxy.
# Creates a local virtualenv on first run, installs deps, then launches.
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating virtualenv in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[setup] installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ ! -f ".env" ]; then
  echo "[setup] no .env found, copying .env.example -> .env (edit it with your keys)"
  cp .env.example .env
fi

echo "[run] starting proxy"
exec python server.py
