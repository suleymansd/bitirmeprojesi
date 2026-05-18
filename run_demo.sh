#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export EVAL_NUM_WORKERS=0
export MODEL_MODE=${MODEL_MODE:-federated}
# export FEDERATED_ROUND=10
uvicorn app.main:app --host 0.0.0.0 --port 8090
