#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-/tmp/lunwen_upload.zip}"
RUN_ROOT="${2:-/root/lunwen_run}"
REPO_DIR="${RUN_ROOT}/lunwen"
LOG_PATH="${RUN_ROOT}/train.log"
PID_PATH="${RUN_ROOT}/train.pid"

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "missing zip: ${ZIP_PATH}" >&2
  exit 2
fi

rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

python3 - "${ZIP_PATH}" "${RUN_ROOT}" <<'PY'
import sys
import zipfile

zip_path = sys.argv[1]
run_root = sys.argv[2]
zipfile.ZipFile(zip_path).extractall(run_root)
print(f"unzipped:{zip_path}->{run_root}")
PY

cd "${REPO_DIR}"
bash scripts/server_prepare_gpu_env.sh "${REPO_DIR}" "${REPO_DIR}/training/configs/remote_a800.yaml"
echo "log_path:${LOG_PATH}"
echo "pid_path:${PID_PATH}"
