#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/lunwen_run/lunwen}"
CONFIG_PATH="${2:-${REPO_DIR}/training/configs/remote_a800.yaml}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "missing repo dir: ${REPO_DIR}" >&2
  exit 2
fi

cd "${REPO_DIR}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

. .venv/bin/activate

export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL="https://pypi.org/simple"
python -m pip install --index-url https://pypi.org/simple --upgrade pip setuptools wheel
python -m pip install --index-url https://pypi.org/simple --upgrade "numpy>=1.26" "scikit-learn>=1.4" "PyYAML>=6.0"

if python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("device0", torch.cuda.get_device_name(0))
PY
  bash scripts/server_start_training.sh "${REPO_DIR}" "$(dirname "${REPO_DIR}")" "${CONFIG_PATH}"
  exit 0
fi

install_torch_cu128() {
  python -m pip install \
    --upgrade \
    --retries 20 \
    --timeout 180 \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.0"
}

python -m pip uninstall -y torch torchvision torchaudio || true

attempt=1
until install_torch_cu128; do
  if [[ "${attempt}" -ge 5 ]]; then
    echo "failed to install torch cu128 after ${attempt} attempts" >&2
    exit 3
  fi
  attempt=$((attempt + 1))
  sleep 10
done

python - <<'PY'
import sys
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda is unavailable after cu128 install")
print("device0", torch.cuda.get_device_name(0))
PY

bash scripts/server_start_training.sh "${REPO_DIR}" "$(dirname "${REPO_DIR}")" "${CONFIG_PATH}"
