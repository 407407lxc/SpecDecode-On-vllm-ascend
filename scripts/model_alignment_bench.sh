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

# Baseline switches: A0/A1/A2
RUN_A0="${RUN_A0:-1}"
RUN_A1="${RUN_A1:-1}"
RUN_A2="${RUN_A2:-1}"

# Fixed K sweep: default 2,4,8
FIXED_K_LIST_STR="${FIXED_K_LIST_STR:-2,4,8}"

# A2 calibration params
A2_ALIGN_SCALE="${A2_ALIGN_SCALE:-1.0}"
A2_ALIGN_BIAS="${A2_ALIGN_BIAS:-0.0}"

# P1/P2 log interval
ALIGN_LOG_INTERVAL="${ALIGN_LOG_INTERVAL:-10}"

# 0 means wait forever until server ready or process exits.
READY_TIMEOUT_S="${READY_TIMEOUT_S:-0}"
# 0 means no benchmark timeout.
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-0}"

# Optional extra args
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

########################
# 2) Environment
########################
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
unset CUDA_VISIBLE_DEVICES || true

LOG_DIR="${LOG_DIR:-./logs/model_alignment_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
SUMMARY_CSV="${LOG_DIR}/summary.csv"
FAILED_TXT="${LOG_DIR}/failed_cases.txt"

CURRENT_PID=""
CURRENT_PGID=""

declare -a CASE_TAGS=()
declare -a CASE_BASELINES=()
declare -a CASE_KS=()
declare -a CASE_FORCE_HS=()
declare -a CASE_ALIGN_ENABLE=()
declare -a CASE_ALIGN_SCALE=()
declare -a CASE_ALIGN_BIAS=()

########################
# 3) Helpers
########################
split_csv() {
  local input="$1"
  local -n out_ref=$2
  out_ref=()
  [[ -z "$input" ]] && return 0
  IFS=',' read -r -a out_ref <<<"$input"
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

add_case() {
  CASE_TAGS+=("$1")
  CASE_BASELINES+=("$2")
  CASE_KS+=("$3")
  CASE_FORCE_HS+=("$4")
  CASE_ALIGN_ENABLE+=("$5")
  CASE_ALIGN_SCALE+=("$6")
  CASE_ALIGN_BIAS+=("$7")
}

build_cases() {
  local ks=()
  split_csv "$FIXED_K_LIST_STR" ks

  for k in "${ks[@]}"; do
    [[ -z "$k" ]] && continue

    if [[ "$RUN_A0" == "1" ]]; then
      add_case "A0_k${k}" "A0" "$k" "0" "0" "1.0" "0.0"
    fi
    if [[ "$RUN_A1" == "1" ]]; then
      add_case "A1_k${k}" "A1" "$k" "1" "0" "1.0" "0.0"
    fi
    if [[ "$RUN_A2" == "1" ]]; then
      add_case "A2_k${k}" "A2" "$k" "1" "1" "$A2_ALIGN_SCALE" "$A2_ALIGN_BIAS"
    fi
  done
}

print_case_table() {
  echo "================ CASES ================"
  local n=${#CASE_TAGS[@]}
  for ((i = 0; i < n; i++)); do
    echo "[$i] tag=${CASE_TAGS[$i]} baseline=${CASE_BASELINES[$i]} k=${CASE_KS[$i]} force_hs=${CASE_FORCE_HS[$i]} align=${CASE_ALIGN_ENABLE[$i]} scale=${CASE_ALIGN_SCALE[$i]} bias=${CASE_ALIGN_BIAS[$i]}"
  done
  echo "======================================="
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

run_bench_cmd() {
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

  if [[ "${BENCH_TIMEOUT_S}" =~ ^[0-9]+$ ]] && ((BENCH_TIMEOUT_S > 0)) && command -v timeout >/dev/null 2>&1; then
    timeout "${BENCH_TIMEOUT_S}s" "${bench_cmd[@]}"
  else
    "${bench_cmd[@]}"
  fi
}

build_case_result_file() {
  local tag="$1"
  local baseline="$2"
  local serve_k="$3"
  local key_log="$4"
  local bench_log="$5"
  local result_log="$6"

  {
    echo "===== CASE ${tag} ====="
    echo "baseline=${baseline}"
    echo "serve_k=${serve_k}"
    echo
    echo "------------ KEY ------------"
    if [[ -s "$key_log" ]]; then
      cat "$key_log"
    else
      echo "(no key lines matched)"
    fi
    echo
    echo "----- BENCHMARK SUMMARY -----"
  } >"$result_log"

  awk '
    /============ Serving Benchmark Result ============/ {in_block=1}
    in_block {print}
    in_block && /^==================================================$/ {exit}
  ' "$bench_log" >>"$result_log"

  if ! grep -q "============ Serving Benchmark Result ============" "$result_log"; then
    echo "(Serving Benchmark Result block not found in bench log)" >>"$result_log"
  fi
}

extract_to_summary_csv() {
  local tag="$1"
  local baseline="$2"
  local serve_k="$3"
  local force_hs="$4"
  local align_enable="$5"
  local align_scale="$6"
  local align_bias="$7"
  local bench_log="$8"
  local server_log="$9"

  python - "$SUMMARY_CSV" "$tag" "$baseline" "$serve_k" "$force_hs" "$align_enable" "$align_scale" "$align_bias" "$bench_log" "$server_log" <<'PY'
import csv
import os
import re
import sys

(summary_csv, tag, baseline, serve_k, force_hs, align_enable, align_scale,
 align_bias, bench_log, server_log) = sys.argv[1:]

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
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

align_matches = re.findall(
    r"ALIGN P1/P2 hs_ratio=([0-9.]+)\s+accept=([0-9.]+)\s+accept_with_hs=([0-9.]+)\s+accept_no_hs=([0-9.]+)\s+throughput_tps=([0-9.]+)\s+iter_p95=([0-9.]+)",
    serv,
)
if align_matches:
    hs_ratio, align_accept, align_accept_with_hs, align_accept_no_hs, align_tps, align_p95_ms = align_matches[-1]
else:
    hs_ratio, align_accept, align_accept_with_hs, align_accept_no_hs, align_tps, align_p95_ms = ("", "", "", "", "", "")

header = [
    "tag", "baseline", "serve_k", "force_hs", "align_enable", "align_scale", "align_bias",
    "successful_requests", "duration_s",
    "output_tok_s", "total_tok_s",
    "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms",
    "p99_ttft_ms", "p99_tpot_ms", "p99_itl_ms",
    "draft_acceptance_rate", "system_efficiency",
    "align_hs_ratio", "align_accept", "align_accept_with_hs", "align_accept_no_hs", "align_tps", "align_p95_ms",
    "bench_log", "server_log",
]
row = [
    tag, baseline, serve_k, force_hs, align_enable, align_scale, align_bias,
    successful_requests, duration_s,
    output_tok_s, total_tok_s,
    mean_ttft_ms, mean_tpot_ms, mean_itl_ms,
    p99_ttft_ms, p99_tpot_ms, p99_itl_ms,
    draft_acceptance_rate, system_efficiency,
    hs_ratio, align_accept, align_accept_with_hs, align_accept_no_hs, align_tps, align_p95_ms,
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
  local baseline="$2"
  local serve_k="$3"
  local force_hs="$4"
  local align_enable="$5"
  local align_scale="$6"
  local align_bias="$7"

  local server_log="${LOG_DIR}/server_${tag}.log"
  local bench_log="${LOG_DIR}/bench_${tag}.log"
  local merged_log="${LOG_DIR}/merged_${tag}.log"
  local key_log="${LOG_DIR}/key_${tag}.log"
  local result_log="${LOG_DIR}/result_${tag}.txt"

  echo "===== CASE ${tag} START $(date '+%F %T') =====" | tee -a "$merged_log"

  kill_stale_vllm

  export VLLM_ASCEND_SPEC_FORCE_RETURN_HS="$force_hs"
  export VLLM_ASCEND_DRAFT_ALIGN_ENABLE="$align_enable"
  export VLLM_ASCEND_DRAFT_ALIGN_SCALE="$align_scale"
  export VLLM_ASCEND_DRAFT_ALIGN_BIAS="$align_bias"
  export VLLM_ASCEND_ALIGN_LOG_INTERVAL="$ALIGN_LOG_INTERVAL"

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

  echo "[INFO] server pid=${CURRENT_PID}, pgid=${CURRENT_PGID}" | tee -a "$merged_log"

  if ! wait_server_ready "$READY_TIMEOUT_S"; then
    tail -n 200 "$server_log" | tee -a "$merged_log" || true
    stop_server
    return 1
  fi

  {
    echo "[INFO] benchmark start $(date '+%F %T')"
    run_bench_cmd
    echo "[INFO] benchmark end $(date '+%F %T')"
  } 2>&1 | tee "$bench_log" | tee -a "$merged_log"

  stop_server

  grep -hE "ALIGN P1/P2|P1P2|Speculative metrics|stage times|Avg generation throughput|Draft acceptance rate|System efficiency" \
    "$server_log" "$bench_log" > "$key_log" || true

  build_case_result_file \
    "$tag" "$baseline" "$serve_k" \
    "$key_log" "$bench_log" "$result_log"

  extract_to_summary_csv \
    "$tag" "$baseline" "$serve_k" \
    "$force_hs" "$align_enable" "$align_scale" "$align_bias" \
    "$bench_log" "$server_log"

  echo "===== CASE ${tag} END $(date '+%F %T') =====" | tee -a "$merged_log"
  echo "[INFO] logs: $server_log | $bench_log | $key_log | $result_log"
}

########################
# 4) Run matrix
########################
build_cases

if [[ ${#CASE_TAGS[@]} -eq 0 ]]; then
  echo "[ERROR] no cases to run. Check RUN_A0/RUN_A1/RUN_A2." >&2
  exit 1
fi

print_case_table | tee "${LOG_DIR}/case_table.txt"

failed=0
: >"$FAILED_TXT"

for ((i = 0; i < ${#CASE_TAGS[@]}; i++)); do
  if ! run_case \
    "${CASE_TAGS[$i]}" \
    "${CASE_BASELINES[$i]}" \
    "${CASE_KS[$i]}" \
    "${CASE_FORCE_HS[$i]}" \
    "${CASE_ALIGN_ENABLE[$i]}" \
    "${CASE_ALIGN_SCALE[$i]}" \
    "${CASE_ALIGN_BIAS[$i]}"; then
    echo "[ERROR] case failed: ${CASE_TAGS[$i]}" | tee -a "$FAILED_TXT"
    failed=$((failed + 1))
  fi
done

echo "[DONE] logs saved to: $LOG_DIR"
echo "[DONE] summary file: $SUMMARY_CSV"

if [[ "$failed" -gt 0 ]]; then
  echo "[DONE] failed cases: $failed (details: $FAILED_TXT)"
  exit 2
fi