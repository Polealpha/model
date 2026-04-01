#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/lunwen_run/lunwen}"
RUN_ROOT="${2:-/root/lunwen_run}"
CONFIG_PATH="${3:-${REPO_DIR}/training/configs/remote_a800.yaml}"
LOG_PATH="${RUN_ROOT}/train.log"
PID_PATH="${RUN_ROOT}/train.pid"

cd "${REPO_DIR}"
nohup bash -lc "set -euo pipefail; export CUDA_VISIBLE_DEVICES=0,1; . .venv/bin/activate && python -c 'import sys, torch; print(\"torch\", torch.__version__); print(\"cuda\", torch.cuda.is_available(), \"count\", torch.cuda.device_count()); sys.exit(0 if torch.cuda.is_available() else 3)' && python scripts/generate_synthetic_dataset.py --config '${CONFIG_PATH}' --force && python scripts/train_structured_baseline.py --config '${CONFIG_PATH}' && python scripts/train_structured_baseline.py --config '${CONFIG_PATH}' --use-weaklabel --model-out training/artifacts/structured_baseline_mixed.pkl --metrics-out reports/structured_baseline_mixed_metrics.json && python scripts/train_multitask_model.py --config '${CONFIG_PATH}' && python scripts/train_multitask_model.py --config '${CONFIG_PATH}' --use-weaklabel --model-out training/artifacts/multitask_mixed_model.pt --metrics-out reports/multitask_mixed_metrics.json && python scripts/train_joint_model.py --config '${CONFIG_PATH}' --use-weaklabel --model-out training/artifacts/joint_model.pt --metrics-out reports/joint_metrics.json && python scripts/build_results_tables.py" > "${LOG_PATH}" 2>&1 < /dev/null &
echo $! > "${PID_PATH}"
echo "started_pid:$(cat "${PID_PATH}")"
echo "log_path:${LOG_PATH}"
