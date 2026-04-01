#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/lunwen_run/lunwen}"
CONFIG_PATH="${2:-${REPO_DIR}/training/configs/remote_a800_trimodal_care.yaml}"
DATA_ROOT="${3:-${REPO_DIR}/datasets/trimodal_emotion}"
PRETRAINED="${4:-${REPO_DIR}/training/artifacts/trimodal_emotion_model.pt}"
MODEL_OUT="${5:-${REPO_DIR}/training/artifacts/trimodal_care_model.pt}"
METRICS_OUT="${6:-${REPO_DIR}/reports/trimodal_care_metrics.json}"

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES=0,1
. .venv/bin/activate

python -u scripts/train_trimodal_care_model.py \
  --config "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --pretrained "${PRETRAINED}" \
  --model-out "${MODEL_OUT}" \
  --metrics-out "${METRICS_OUT}"
