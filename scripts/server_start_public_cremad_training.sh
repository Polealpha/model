#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/lunwen_run/lunwen}"
PUBLIC_ROOT="${2:-/root/datasets_public}"
CREMAD_DIR="${PUBLIC_ROOT}/crema-d-mirror"
DATA_ROOT="${REPO_DIR}/datasets/public_cremad_trimodal"
CONFIG_PATH="${3:-${REPO_DIR}/training/configs/public_cremad_trimodal.yaml}"
MODEL_OUT="${4:-${REPO_DIR}/training/artifacts/public_cremad_trimodal_emotion_model.pt}"
METRICS_OUT="${5:-${REPO_DIR}/reports/public_cremad_trimodal_emotion_metrics.json}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git-lfs
git lfs install

mkdir -p "${PUBLIC_ROOT}"
if [[ ! -d "${CREMAD_DIR}/.git" ]]; then
  git clone https://gitlab.com/cs-cooper-lab/crema-d-mirror.git "${CREMAD_DIR}"
fi

cd "${CREMAD_DIR}"
git lfs pull

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES=0,1
. .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple --upgrade opencv-python-headless
python -u scripts/build_cremad_trimodal_dataset.py --repo-root "${CREMAD_DIR}" --output-root "${DATA_ROOT}" --sequence-steps 48
python -u scripts/train_trimodal_emotion_model.py \
  --config "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --model-out "${MODEL_OUT}" \
  --metrics-out "${METRICS_OUT}"
