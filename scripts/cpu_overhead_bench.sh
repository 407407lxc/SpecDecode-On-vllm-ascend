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
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8B}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-16}"

RUN_C0="${RUN_C0:-1}"
RUN_C1="${RUN_C1:-1}"
RUN_C2="${RUN_C2:-1}"

C0_K="${C0_K:-1}"
C1_K="${C1_K:-8}"
C2_K="${C2_K:-8}"

# Optional per-case bench args (for example, set logprobs request fields).
C0_BENCH_EXTRA_ARGS="${C0_BENCH_EXTRA_ARGS:-}"
C1_BENCH_EXTRA_ARGS="${C1_BENCH_EXTRA_ARGS:-}"
C2_BENCH_EXTRA_ARGS="${C2_BENCH_EXTRA_ARGS:-}"

MS_STATS_INTERVAL="${MS_STATS_INTERVAL:-200}"
ALIGN_LOG_INTERVAL="${ALIGN_LOG_INTERVAL:-200}"

READY_TIMEOUT_S="${READY_TIMEOUT_S:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-0}"

SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

########################
# 2) Environment
########################
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
unset CUDA_VISIBLE_DEVICES || true

LOG_DIR="${LOG_DIR:-./logs/cpu_overhead_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
SUMMARY_CSV="${LOG_DIR}/summary.csv"

CURRENT_PID=""
CURRENT_PGID=""

declare -a CASE_TAGS=()
declare -a CASE_KS=()
declare -a CASE_LOGPROBS_MODE=()
declare -a CASE_BENCH_EXTRA=()

########################
# 3) Helpers
########################
add_case() {
  CASE_TAGS+=("$1")
  CASE_KS+=("$2")
  CASE_LOGPROBS_MODE+=("$3")
  CASE_BENCH_EXTRA+=("$4")
}

build_cases() {
  if [[ "$RUN_C0" == "1" ]]; then
    add_case "C0_k${C0_K}_logprobs_off" "$C0_K" "off" "$C0_BENCH_EXTRA_ARGS"
  fi
  if [[ "$RUN_C1" == "1" ]]; then
    add_case "C1_k${C1_K}_logprobs_on" "$C1_K" "on" "$C1_BENCH_EXTRA_ARGS"
  fi
  if [[ "$RUN_C2" == "1" ]]; then
    add_case "C2_k${C2_K}_logprobs_off" "$C2_K" "off" "$C2_BENCH_EXTRA_ARGS"
  fi
}

append_space_split_args() {
  local arg_string="$1"
  local -n target_arr=$2
  [[ -z "$arg_string" ]] && return 0
  local extra=()
  # shellcheck disable=SC2206
  extra=($arg_string)
  target_arr+=("${extra[@]}")
}

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
  local timeout_s="${1:-0}"
  local i=0
  while true; do
    if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[INFO] server ready after ${i}s"
      return 0
    fi
    if [[ -n "${CURRENT_PID:-}" ]] && ! kill -0 "${CURRENT_PID}" 2>/dev/null; then
      echo "[ERROR] server exited unexpectedly"
      return 1
    fi
    sleep 1
    ((i += 1))

    if [[ "${timeout_s}" =~ ^[0-9]+$ ]] && ((timeout_s > 0)) && ((i >= timeout_s)); then
      echo "[ERROR] server not ready in ${timeout_s}s"
      return 1
    fi
  done
}

extract_to_summary_csv() {
  local tag="$1"
  local serve_k="$2"
  local logprobs_mode="$3"
  local bench_log="$4"
  local server_log="$5"

  python - "$SUMMARY_CSV" "$tag" "$serve_k" "$logprobs_mode" "$bench_log" "$server_log" <<'PY'
import csv
import os
import re
import sys

summary_csv, tag, serve_k, logprobs_mode, bench_log, server_log = sys.argv[1:]

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except FileNotFoundError:
        return ""

bench = read_text(bench_log)
serv = read_text(server_log)

def find_num(text, label):
    m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|", text, re.I)
    if m:
        return m.group(1)
    m = re.search(rf"{re.escape(label)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    return m.group(1) if m else ""

output_tok_s = find_num(bench, "Output token throughput (tok/s)")
total_tok_s = find_num(bench, "Total token throughput (tok/s)")
mean_tpot_ms = find_num(bench, "Mean TPOT (ms)")
mean_itl_ms = find_num(bench, "Mean ITL (ms)")
p99_tpot_ms = find_num(bench, "P99 TPOT (ms)")
p99_itl_ms = find_num(bench, "P99 ITL (ms)")

draft = re.findall(r"Draft acceptance rate:\s*([0-9.]+),\s*System efficiency:\s*([0-9.]+)", serv)
if draft:
    draft_acceptance_rate, system_efficiency = draft[-1]
else:
    draft_acceptance_rate, system_efficiency = "", ""

fastpath = re.findall(
    r"DraftGPUFastPath hit_rate=([0-9.]+) total=([0-9]+) hit=([0-9]+) fail_disabled=([0-9]+) fail_prompt=([0-9]+) fail_backend=([0-9]+) fail_lora=([0-9]+) fail_adapter=([0-9]+)",
    serv,
)
if fastpath:
    ms_hit_rate, ms_total, ms_hit, ms_fail_disabled, ms_fail_prompt, ms_fail_backend, ms_fail_lora, ms_fail_adapter = fastpath[-1]
else:
    ms_hit_rate, ms_total, ms_hit, ms_fail_disabled, ms_fail_prompt, ms_fail_backend, ms_fail_lora, ms_fail_adapter = ("", "", "", "", "", "", "", "")

header = [
    "tag", "serve_k", "logprobs_mode",
    "output_tok_s", "total_tok_s", "mean_tpot_ms", "mean_itl_ms", "p99_tpot_ms", "p99_itl_ms",
    "draft_acceptance_rate", "system_efficiency",
    "ms_hit_rate", "ms_total", "ms_hit", "ms_fail_disabled", "ms_fail_prompt", "ms_fail_backend", "ms_fail_lora", "ms_fail_adapter",
    "bench_log", "server_log",
]
row = [
    tag, serve_k, logprobs_mode,
    output_tok_s, total_tok_s, mean_tpot_ms, mean_itl_ms, p99_tpot_ms, p99_itl_ms,
    draft_acceptance_rate, system_efficiency,
    ms_hit_rate, ms_total, ms_hit, ms_fail_disabled, ms_fail_prompt, ms_fail_backend, ms_fail_lora, ms_fail_adapter,
    bench_log, server_log,
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
  local tag="$1"
  local serve_k="$2"
  local logprobs_mode="$3"
  local bench_extra="$4"

  local server_log="${LOG_DIR}/server_${tag}.log"
  local bench_log="${LOG_DIR}/bench_${tag}.log"

  echo "===== CASE ${tag} START $(date '+%F %T') ====="
  kill_stale_vllm

  export VLLM_LOGPROBS_MODE="$logprobs_mode"
  export VLLM_ASCEND_MS_STATS_INTERVAL="$MS_STATS_INTERVAL"
  export VLLM_ASCEND_ALIGN_LOG_INTERVAL="$ALIGN_LOG_INTERVAL"
  export VLLM_ASCEND_DRAFT_ALIGN_ENABLE="0"

  local -a serve_cmd=(
    vllm serve "$MODEL_PATH"
    --gpu-memory-utilization 0.90
    --max-model-len 8192
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --port "$PORT"
    --host "$HOST"
    --enforce-eager
  )
  append_space_split_args "$SERVE_EXTRA_ARGS" serve_cmd

  local spec_cfg
  printf -v spec_cfg '{"method":"draft_model","model":"%s","num_speculative_tokens":%d}' \
    "$DRAFT_MODEL_PATH" "$serve_k"
  serve_cmd+=(--speculative-config "$spec_cfg")

  setsid "${serve_cmd[@]}" >"$server_log" 2>&1 &
  CURRENT_PID=$!
  CURRENT_PGID="$(ps -o pgid= -p "$CURRENT_PID" | tr -d ' ' || true)"
  [[ -z "${CURRENT_PGID}" ]] && CURRENT_PGID="$CURRENT_PID"

  if ! wait_server_ready "$READY_TIMEOUT_S"; then
    tail -n 200 "$server_log" || true
    stop_server
    return 1
  fi

  local -a bench_cmd=(
    vllm bench serve
    --model "$SERVED_MODEL_NAME"
    --tokenizer "$MODEL_PATH"
    --base-url "http://${HOST}:${PORT}"
    --endpoint-type vllm
    --dataset-name sharegpt
    --dataset-path "$DATASET_PATH"
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$REQUEST_RATE"
    --trust-remote-code
  )
  append_space_split_args "$BENCH_EXTRA_ARGS" bench_cmd
  append_space_split_args "$bench_extra" bench_cmd

  if [[ "${BENCH_TIMEOUT_S}" =~ ^[0-9]+$ ]] && ((BENCH_TIMEOUT_S > 0)) && command -v timeout >/dev/null 2>&1; then
    timeout "${BENCH_TIMEOUT_S}s" "${bench_cmd[@]}" 2>&1 | tee "$bench_log"
  else
    "${bench_cmd[@]}" 2>&1 | tee "$bench_log"
  fi

  stop_server

  extract_to_summary_csv "$tag" "$serve_k" "$logprobs_mode" "$bench_log" "$server_log"

  echo "===== CASE ${tag} END $(date '+%F %T') ====="
}

########################
# 4) Run matrix
########################
build_cases

if [[ ${#CASE_TAGS[@]} -eq 0 ]]; then
  echo "[ERROR] no cases to run. Check RUN_C0/RUN_C1/RUN_C2." >&2
  exit 1
fi

for ((i = 0; i < ${#CASE_TAGS[@]}; i++)); do
  run_case "${CASE_TAGS[$i]}" "${CASE_KS[$i]}" "${CASE_LOGPROBS_MODE[$i]}" "${CASE_BENCH_EXTRA[$i]}"
done

echo "[DONE] logs saved to: $LOG_DIR"
echo "[DONE] summary file: $SUMMARY_CSV"
