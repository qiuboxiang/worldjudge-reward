#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-${ROOT}/.venv}"
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to the local Qwen3-VL-4B-Instruct directory}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the local WorldJudge checkpoint directory}"
VIDEO_ROOT="${ROOT}/demo_data/lerobot"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/ckpt_21500}"

for path in "${VENV}/bin/python" "${ROOT}/worldjudge_runtime" "${BASE_MODEL}" "${CHECKPOINT}" "${VIDEO_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done

for demo in 5638 6005 5645; do
  if [[ ! -f "${ROOT}/demos/${demo}.jsonl" ]]; then
    echo "Missing demo annotation: ${ROOT}/demos/${demo}.jsonl" >&2
    exit 2
  fi
done

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${VENV}/bin/python" -I "${ROOT}/infer.py" \
    --base-model "${BASE_MODEL}" \
    --checkpoint "${CHECKPOINT}" \
    --video-root "${VIDEO_ROOT}" \
    --demo 5638="${ROOT}/demos/5638.jsonl" \
    --demo 6005="${ROOT}/demos/6005.jsonl" \
    --demo 5645="${ROOT}/demos/5645.jsonl" \
    --output-dir "${OUTPUT_DIR}"

"${VENV}/bin/python" "${ROOT}/verify_outputs.py" "${OUTPUT_DIR}/summary.json"
