#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs

SERVER_ADDRESS=${SERVER_ADDRESS:-0.0.0.0:8092}
CLIENT_SERVER_ADDRESS=${CLIENT_SERVER_ADDRESS:-127.0.0.1:8092}
NUM_ROUNDS=${NUM_ROUNDS:-20}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-64}
export FEDERATED_CKPT_DIR=${FEDERATED_CKPT_DIR:-checkpoints/federated_improved_full}
export LOSS_MODE=${LOSS_MODE:-focal}
mkdir -p "$FEDERATED_CKPT_DIR"

# önce eski süreçleri temizle
pkill -f "src/server.py" >/dev/null 2>&1 || true
pkill -f "src/client.py --client_id 1" >/dev/null 2>&1 || true
pkill -f "src/client.py --client_id 2" >/dev/null 2>&1 || true

SERVER_ADDRESS="$SERVER_ADDRESS" NUM_ROUNDS="$NUM_ROUNDS" \
  nohup python -u src/server.py > logs/server_tuning.log 2>&1 &
sleep 3
nohup python -u src/client.py --client_id 1 \
  --server_address "$CLIENT_SERVER_ADDRESS" \
  --local_epochs "$LOCAL_EPOCHS" \
  --batch_size "$BATCH_SIZE" > logs/client1_tuning.log 2>&1 &
nohup python -u src/client.py --client_id 2 \
  --server_address "$CLIENT_SERVER_ADDRESS" \
  --local_epochs "$LOCAL_EPOCHS" \
  --batch_size "$BATCH_SIZE" > logs/client2_tuning.log 2>&1 &

echo "Tuning başlatıldı."
echo "Checkpoint dizini: $FEDERATED_CKPT_DIR"
echo "Server: $SERVER_ADDRESS | Client: $CLIENT_SERVER_ADDRESS"
echo "Rounds: $NUM_ROUNDS | Local epochs: $LOCAL_EPOCHS | Batch size: $BATCH_SIZE | Loss: $LOSS_MODE"
echo "Loglar: logs/server_tuning.log, logs/client1_tuning.log, logs/client2_tuning.log"
