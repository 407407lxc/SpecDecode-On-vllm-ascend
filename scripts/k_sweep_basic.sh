#!/usr/bin/env bash
set -euo pipefail

########################
# 1) Basic config
########################
MODEL_PATH="${MODEL_PATH:-/mnt/ky2307909/siyuan.tong/Qwen3-8B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/mnt/ky2307909/siyuan.tong/Qwen3-0-6B}"
DATASET_PATH="${DATASET_PATH:-/mnt/ky2307909/siyuan.tong/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-16}"

# Fixed K sweep: 2,4,8
FIXED_K_LIST=(2 4 8)

########################
# 2) Environment
########################
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
unset CUDA_VISIBLE_DEVICES || true

LOG_DIR="${LOG_DIR:-./logs/spec_fixed_k_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

SUMMARY_CSV="${LOG_DIR}/summary.csv"

CURRENT_PID=""
CURRENT_PGID=""

########################
# 3) Helpers
########################
stop_server() {
  set +e
  if [[ -n "${CURRENT_PGID:-}" ]]; then
    kill -TERM -- "-${CURRENT_PGID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${CURRENT_PGID}" 2>/dev/null || true
  fi
  if [[ -n "${CURRENT_PID:-}" ]]; then
    kill -KILL "${CURRENT_PID}" 2>/dev/null || true
  fi
  CURRENT_PID=""
  CURRENT_PGID=""
  set -e
}

cleanup() { stop_server; }
trap cleanup EXIT INT TERM

kill_stale_vllm() {
  set +e
  pkill -TERM -f "vllm serve|api_server.py|engine.py|multiprocessing.engine" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "vllm serve|api_server.py|engine.py|multiprocessing.engine" 2>/dev/null || true
  set -e
}

wait_server_ready() {
  local timeout_s="${1:-180}"
  local i=0
  while (( i < timeout_s )); do
    if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[INFO] server ready after ${i}s"
      return 0
    fi
    if [[ -n "${CURRENT_PID:-}" ]] && ! kill -0 "${CURRENT_PID}" 2>/dev/null; then
      echo "[ERROR] server exited unexpectedly"
      return 1
    fi
    sleep 1
    ((i+=1))
  done
  echo "[ERROR] server not ready in ${timeout_s}s"
  return 1
}

extract_to_summary_csv() {
  local tag="$1"
  local serve_k="$2"
  local bench_log="$3"
  local server_log="$4"

  python - "$SUMMARY_CSV" "$tag" "$serve_k" "$bench_log" "$server_log" <<'PY'
import csv
import os
import re
import sys

summary_csv, tag, serve_k, bench_log, server_log = sys.argv[1:]

def read_text(p):
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except FileNotFoundError:
        return ""

bench = read_text(bench_log)
serv = read_text(server_log)

def find_num_from_table_or_text(text, label):
    m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|", text, re.I)
    if m:
        return m.group(1)
    m = re.search(rf"{re.escape(label)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if m:
        return m.group(1)
    return ""

successful_requests = find_num_from_table_or_text(bench, "Successful requests")
duration_s = find_num_from_table_or_text(bench, "Benchmark duration (s)")
output_tok_s = find_num_from_table_or_text(bench, "Output token throughput (tok/s)")
total_tok_s = find_num_from_table_or_text(bench, "Total token throughput (tok/s)")
mean_ttft_ms = find_num_from_table_or_text(bench, "Mean TTFT (ms)")
mean_tpot_ms = find_num_from_table_or_text(bench, "Mean TPOT (ms)")
mean_itl_ms = find_num_from_table_or_text(bench, "Mean ITL (ms)")
p99_ttft_ms = find_num_from_table_or_text(bench, "P99 TTFT (ms)")
p99_tpot_ms = find_num_from_table_or_text(bench, "P99 TPOT (ms)")
p99_itl_ms = find_num_from_table_or_text(bench, "P99 ITL (ms)")

spec_matches = re.findall(
    r"Draft acceptance rate:\s*([0-9.]+),\s*System efficiency:\s*([0-9.]+)",
    serv
)
if spec_matches:
    draft_acceptance_rate, system_efficiency = spec_matches[-1]
else:
    draft_acceptance_rate, system_efficiency = "", ""

header = [
    "tag", "serve_k",
    "successful_requests", "duration_s",
    "output_tok_s", "total_tok_s",
    "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms",
    "p99_ttft_ms", "p99_tpot_ms", "p99_itl_ms",
    "draft_acceptance_rate", "system_efficiency",
    "bench_log", "server_log"
]
row = [
    tag, serve_k,
    successful_requests, duration_s,
    output_tok_s, total_tok_s,
    mean_ttft_ms, mean_tpot_ms, mean_itl_ms,
    p99_ttft_ms, p99_tpot_ms, p99_itl_ms,
    draft_acceptance_rate, system_efficiency,
    bench_log, server_log
]

need_header = not os.path.exists(summary_csv)
with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if need_header:
        w.writerow(header)
    w.writerow(row)
PY
}

run_case() {
  local TAG="$1"
  local SERVE_K="$2"

  local SERVER_LOG="${LOG_DIR}/server_${TAG}.log"
  local BENCH_LOG="${LOG_DIR}/bench_${TAG}.log"
  local MERGED_LOG="${LOG_DIR}/merged_${TAG}.log"
  local KEY_LOG="${LOG_DIR}/key_${TAG}.log"

  echo "===== CASE ${TAG} START $(date '+%F %T') =====" | tee -a "$MERGED_LOG"

  kill_stale_vllm

  local -a SERVE_CMD=(
    vllm serve "$MODEL_PATH"
    --gpu-memory-utilization 0.90
    --max-model-len 8192
    --served-model-name qwen3-8B
    --trust-remote-code
    --port "$PORT"
    --host "$HOST"
    --enforce-eager
  )

  local SPEC_CFG
  printf -v SPEC_CFG '{"method":"draft_model","model":"%s","num_speculative_tokens":%d}' \
    "$DRAFT_MODEL_PATH" "$SERVE_K"
  SERVE_CMD+=(--speculative-config "$SPEC_CFG")

  setsid "${SERVE_CMD[@]}" >"$SERVER_LOG" 2>&1 &
  CURRENT_PID=$!
  CURRENT_PGID="$(ps -o pgid= -p "$CURRENT_PID" | tr -d ' ' || true)"
  [[ -z "${CURRENT_PGID}" ]] && CURRENT_PGID="$CURRENT_PID"

  echo "[INFO] server pid=${CURRENT_PID}, pgid=${CURRENT_PGID}" | tee -a "$MERGED_LOG"

  if ! wait_server_ready 180; then
    tail -n 200 "$SERVER_LOG" | tee -a "$MERGED_LOG" || true
    stop_server
    return 1
  fi

  {
    echo "[INFO] benchmark start $(date '+%F %T')"
    vllm bench serve \
      --model qwen3-8B \
      --tokenizer "$MODEL_PATH" \
      --base-url "http://${HOST}:${PORT}" \
      --endpoint-type vllm \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_PATH" \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate "$REQUEST_RATE" \
      --trust-remote-code
    echo "[INFO] benchmark end $(date '+%F %T')"
  } 2>&1 | tee "$BENCH_LOG" | tee -a "$MERGED_LOG"

  stop_server

  grep -E "Speculative metrics|stage times|Avg generation throughput|Draft acceptance rate|System efficiency" \
    "$SERVER_LOG" "$BENCH_LOG" > "$KEY_LOG" || true

  extract_to_summary_csv "$TAG" "$SERVE_K" "$BENCH_LOG" "$SERVER_LOG"

  echo "===== CASE ${TAG} END $(date '+%F %T') =====" | tee -a "$MERGED_LOG"
  echo "[INFO] logs: $SERVER_LOG | $BENCH_LOG | $KEY_LOG"
}

########################
# 4) Run fixed K matrix
########################
for K in "${FIXED_K_LIST[@]}"; do
  run_case "k${K}_fixed" "$K"
done

echo "[DONE] logs saved to: $LOG_DIR"
echo "[DONE] summary file: $SUMMARY_CSV"
