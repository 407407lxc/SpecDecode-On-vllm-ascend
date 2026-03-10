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
A2_ALIGN_MODE="${A2_ALIGN_MODE:-${VLLM_ASCEND_DRAFT_ALIGN_MODE:-temperature}}"
A2_ALIGN_TEMPERATURE="${A2_ALIGN_TEMPERATURE:-${VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE:-1.0}}"
A2_ALIGN_VOCAB_BIAS_PATH="${A2_ALIGN_VOCAB_BIAS_PATH:-${VLLM_ASCEND_DRAFT_ALIGN_VOCAB_BIAS_PATH:-}}"

# P1/P2 log interval
ALIGN_LOG_INTERVAL="${ALIGN_LOG_INTERVAL:-10}"

# 0 means wait forever until server ready or process exits.
READY_TIMEOUT_S="${READY_TIMEOUT_S:-0}"
# 0 means no benchmark timeout.
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-0}"

# Optional extra args
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
# Log file mode: full (legacy) | two (only server.log + bench.log)
LOG_FILE_MODE="${LOG_FILE_MODE:-full}"
# When LOG_FILE_MODE=two: 0=truncate logs, 1=append to existing logs.
LOG_FILE_APPEND="${LOG_FILE_APPEND:-0}"

########################
# 2) Environment
########################
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
unset CUDA_VISIBLE_DEVICES || true

LOG_DIR="${LOG_DIR:-./logs/model_alignment_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

if [[ "$LOG_FILE_MODE" != "full" && "$LOG_FILE_MODE" != "two" ]]; then
  echo "[ERROR] unsupported LOG_FILE_MODE=$LOG_FILE_MODE (use full|two)" >&2
  exit 1
fi

if [[ "$LOG_FILE_MODE" == "two" ]]; then
  SERVER_LOG_FILE="${LOG_DIR}/server.log"
  BENCH_LOG_FILE="${LOG_DIR}/bench.log"
  if [[ "$LOG_FILE_APPEND" == "1" ]]; then
    touch "$SERVER_LOG_FILE" "$BENCH_LOG_FILE"
  else
    : >"$SERVER_LOG_FILE"
    : >"$BENCH_LOG_FILE"
  fi
  SUMMARY_CSV=""
  FAILED_TXT=""
else
  SUMMARY_CSV="${LOG_DIR}/summary.csv"
  FAILED_TXT="${LOG_DIR}/failed_cases.txt"
fi

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
    echo "[$i] tag=${CASE_TAGS[$i]} baseline=${CASE_BASELINES[$i]} k=${CASE_KS[$i]} force_hs=${CASE_FORCE_HS[$i]} align=${CASE_ALIGN_ENABLE[$i]} mode=${A2_ALIGN_MODE} temp=${A2_ALIGN_TEMPERATURE} scale=${CASE_ALIGN_SCALE[$i]} bias=${CASE_ALIGN_BIAS[$i]} vocab_bias=${A2_ALIGN_VOCAB_BIAS_PATH}"
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
  local align_mode="$8"
  local align_temperature="$9"
  local align_vocab_bias_path="${10}"
  local adaptive_k_enable="${11}"
  local adaptive_enable_utility="${12}"
  local adaptive_align_gate_enable="${13}"
  local bench_log="${14}"
  local server_log="${15}"

  python - "$SUMMARY_CSV" "$tag" "$baseline" "$serve_k" "$force_hs" "$align_enable" "$align_scale" "$align_bias" "$align_mode" "$align_temperature" "$align_vocab_bias_path" "$adaptive_k_enable" "$adaptive_enable_utility" "$adaptive_align_gate_enable" "$bench_log" "$server_log" <<'PY'
import csv
import os
import re
import sys

(summary_csv, tag, baseline, serve_k, force_hs, align_enable, align_scale,
 align_bias, align_mode, align_temperature, align_vocab_bias_path,
 adaptive_k_enable, adaptive_enable_utility, adaptive_align_gate_enable,
 bench_log, server_log) = sys.argv[1:]


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except FileNotFoundError:
        return ""


bench = read_text(bench_log)
serv = read_text(server_log)


def find_num_from_table_or_text(text, label):
    m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|",
                  text, re.I)
    if m:
        return m.group(1)
    m = re.search(rf"{re.escape(label)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                  text, re.I)
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
    serv,
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

align_cfg_matches = re.findall(
    r"DRAFT_ALIGN config:\s*enable=([^\s]+)\s+mode=([^\s]+)\s+temperature=([0-9.]+)\s+scale=([-0-9.]+)\s+bias=([-0-9.]+)\s+vocab_bias_path=([^\s]+)\s+supports_previous_hidden_states=([^\s]+)",
    serv,
)
if align_cfg_matches:
    (runner_align_enable, runner_align_mode, runner_align_temperature,
     runner_align_scale, runner_align_bias, runner_vocab_bias_path,
     supports_previous_hidden_states) = align_cfg_matches[-1]
else:
    runner_align_enable = align_enable
    runner_align_mode = align_mode
    runner_align_temperature = align_temperature
    runner_align_scale = align_scale
    runner_align_bias = align_bias
    runner_vocab_bias_path = align_vocab_bias_path or ""
    supports_previous_hidden_states = ""

adaptive_stats_matches = re.findall(
    r"AdaptiveK stats switch_per_min=([0-9.]+)\s+high_k_occ=([0-9.]+)\s+hist=([^\s]+)",
    serv,
)
if adaptive_stats_matches:
    adaptive_switch_per_min, adaptive_high_k_occ, adaptive_hist = adaptive_stats_matches[-1]
else:
    adaptive_switch_per_min, adaptive_high_k_occ, adaptive_hist = "", "", ""

adaptive_gate_matches = re.findall(
    r"AdaptiveK align_gate score=([0-9.]+)\s+delta_hs=([-0-9.]+)\s+cap=([^\s]+)",
    serv,
)
if adaptive_gate_matches:
    adaptive_align_score, adaptive_delta_hs, adaptive_align_cap = adaptive_gate_matches[-1]
else:
    adaptive_align_score, adaptive_delta_hs, adaptive_align_cap = "", "", ""

header = [
    "tag", "baseline", "serve_k", "force_hs",
    "align_enable", "align_mode", "align_temperature", "align_scale", "align_bias", "align_vocab_bias_path",
    "adaptive_k_enable", "adaptive_enable_utility", "adaptive_align_gate_enable",
    "successful_requests", "duration_s",
    "output_tok_s", "total_tok_s",
    "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms",
    "p99_ttft_ms", "p99_tpot_ms", "p99_itl_ms",
    "draft_acceptance_rate", "system_efficiency",
    "align_hs_ratio", "align_accept", "align_accept_with_hs", "align_accept_no_hs", "align_tps", "align_p95_ms",
    "runner_align_enable", "runner_align_mode", "runner_align_temperature", "runner_align_scale", "runner_align_bias", "runner_vocab_bias_path", "supports_previous_hidden_states",
    "adaptive_switch_per_min", "adaptive_high_k_occ", "adaptive_hist",
    "adaptive_align_score", "adaptive_delta_hs", "adaptive_align_cap",
    "bench_log", "server_log",
]
row = [
    tag, baseline, serve_k, force_hs,
    align_enable, align_mode, align_temperature, align_scale, align_bias, align_vocab_bias_path,
    adaptive_k_enable, adaptive_enable_utility, adaptive_align_gate_enable,
    successful_requests, duration_s,
    output_tok_s, total_tok_s,
    mean_ttft_ms, mean_tpot_ms, mean_itl_ms,
    p99_ttft_ms, p99_tpot_ms, p99_itl_ms,
    draft_acceptance_rate, system_efficiency,
    hs_ratio, align_accept, align_accept_with_hs, align_accept_no_hs, align_tps, align_p95_ms,
    runner_align_enable, runner_align_mode, runner_align_temperature, runner_align_scale, runner_align_bias, runner_vocab_bias_path, supports_previous_hidden_states,
    adaptive_switch_per_min, adaptive_high_k_occ, adaptive_hist,
    adaptive_align_score, adaptive_delta_hs, adaptive_align_cap,
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
  local align_mode="${A2_ALIGN_MODE}"
  local align_temperature="${A2_ALIGN_TEMPERATURE}"
  local align_vocab_bias_path="${A2_ALIGN_VOCAB_BIAS_PATH}"
  local adaptive_k_enable="${VLLM_ASCEND_ADAPTIVE_K_ENABLE:-0}"
  local adaptive_enable_utility="${VLLM_ASCEND_ADAPTIVE_ENABLE_UTILITY:-0}"
  local adaptive_align_gate_enable="${VLLM_ASCEND_ADAPTIVE_ALIGN_GATE_ENABLE:-0}"

  local server_log=""
  local bench_log=""
  local merged_log=""
  local key_log=""
  local result_log=""

  if [[ "$LOG_FILE_MODE" == "two" ]]; then
    server_log="$SERVER_LOG_FILE"
    bench_log="$BENCH_LOG_FILE"
    {
      echo "===== CASE ${tag} START $(date '+%F %T') ====="
      echo "baseline=${baseline} serve_k=${serve_k} force_hs=${force_hs} align=${align_enable} mode=${align_mode} temp=${align_temperature} scale=${align_scale} bias=${align_bias}"
    } >>"$server_log"
    {
      echo "===== CASE ${tag} START $(date '+%F %T') ====="
      echo "baseline=${baseline} serve_k=${serve_k} force_hs=${force_hs} align=${align_enable} mode=${align_mode} temp=${align_temperature} scale=${align_scale} bias=${align_bias}"
    } >>"$bench_log"
  else
    server_log="${LOG_DIR}/server_${tag}.log"
    bench_log="${LOG_DIR}/bench_${tag}.log"
    merged_log="${LOG_DIR}/merged_${tag}.log"
    key_log="${LOG_DIR}/key_${tag}.log"
    result_log="${LOG_DIR}/result_${tag}.txt"
    echo "===== CASE ${tag} START $(date '+%F %T') =====" | tee -a "$merged_log"
  fi

  kill_stale_vllm

  export VLLM_ASCEND_SPEC_FORCE_RETURN_HS="$force_hs"
  export VLLM_ASCEND_DRAFT_ALIGN_ENABLE="$align_enable"
  export VLLM_ASCEND_DRAFT_ALIGN_MODE="$align_mode"
  export VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE="$align_temperature"
  export VLLM_ASCEND_DRAFT_ALIGN_SCALE="$align_scale"
  export VLLM_ASCEND_DRAFT_ALIGN_BIAS="$align_bias"
  export VLLM_ASCEND_DRAFT_ALIGN_VOCAB_BIAS_PATH="$align_vocab_bias_path"
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

  if [[ "$LOG_FILE_MODE" == "two" ]]; then
    setsid "${serve_cmd[@]}" >>"$server_log" 2>&1 &
  else
    setsid "${serve_cmd[@]}" >"$server_log" 2>&1 &
  fi
  CURRENT_PID=$!
  CURRENT_PGID="$(ps -o pgid= -p "$CURRENT_PID" | tr -d ' ' || true)"
  [[ -z "${CURRENT_PGID}" ]] && CURRENT_PGID="$CURRENT_PID"

  if [[ "$LOG_FILE_MODE" == "two" ]]; then
    echo "[INFO] server pid=${CURRENT_PID}, pgid=${CURRENT_PGID}" >>"$server_log"
    echo "[INFO] server pid=${CURRENT_PID}, pgid=${CURRENT_PGID}" >>"$bench_log"
  else
    echo "[INFO] server pid=${CURRENT_PID}, pgid=${CURRENT_PGID}" | tee -a "$merged_log"
  fi

  if ! wait_server_ready "$READY_TIMEOUT_S"; then
    if [[ "$LOG_FILE_MODE" == "two" ]]; then
      tail -n 200 "$server_log" || true
    else
      tail -n 200 "$server_log" | tee -a "$merged_log" || true
    fi
    stop_server
    return 1
  fi

  if [[ "$LOG_FILE_MODE" == "two" ]]; then
    {
      echo "[INFO] benchmark start $(date '+%F %T')"
      run_bench_cmd
      echo "[INFO] benchmark end $(date '+%F %T')"
    } 2>&1 | tee -a "$bench_log"
  else
    {
      echo "[INFO] benchmark start $(date '+%F %T')"
      run_bench_cmd
      echo "[INFO] benchmark end $(date '+%F %T')"
    } 2>&1 | tee "$bench_log" | tee -a "$merged_log"
  fi

  stop_server

  if [[ "$LOG_FILE_MODE" == "two" ]]; then
    echo "===== CASE ${tag} END $(date '+%F %T') =====" >>"$server_log"
    echo "===== CASE ${tag} END $(date '+%F %T') =====" >>"$bench_log"
    echo "[INFO] logs: $server_log | $bench_log"
    return 0
  fi

  grep -hE "ALIGN P1/P2|P1P2|AdaptiveK update|AdaptiveK utility switch|AdaptiveK stats|AdaptiveK align_gate|DRAFT_ALIGN config|Speculative metrics|stage times|Avg generation throughput|Draft acceptance rate|System efficiency" \
    "$server_log" "$bench_log" > "$key_log" || true

  build_case_result_file \
    "$tag" "$baseline" "$serve_k" \
    "$key_log" "$bench_log" "$result_log"

  extract_to_summary_csv \
    "$tag" "$baseline" "$serve_k" \
    "$force_hs" "$align_enable" "$align_scale" "$align_bias" \
    "$align_mode" "$align_temperature" "$align_vocab_bias_path" \
    "$adaptive_k_enable" "$adaptive_enable_utility" "$adaptive_align_gate_enable" \
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

if [[ "$LOG_FILE_MODE" == "two" ]]; then
  print_case_table
else
  print_case_table | tee "${LOG_DIR}/case_table.txt"
fi

failed=0
declare -a FAILED_CASES=()
if [[ "$LOG_FILE_MODE" == "full" ]]; then
  : >"$FAILED_TXT"
fi

for ((i = 0; i < ${#CASE_TAGS[@]}; i++)); do
  if ! run_case \
    "${CASE_TAGS[$i]}" \
    "${CASE_BASELINES[$i]}" \
    "${CASE_KS[$i]}" \
    "${CASE_FORCE_HS[$i]}" \
    "${CASE_ALIGN_ENABLE[$i]}" \
    "${CASE_ALIGN_SCALE[$i]}" \
    "${CASE_ALIGN_BIAS[$i]}"; then
    msg="[ERROR] case failed: ${CASE_TAGS[$i]}"
    echo "$msg"
    FAILED_CASES+=("${CASE_TAGS[$i]}")
    if [[ "$LOG_FILE_MODE" == "full" ]]; then
      echo "$msg" | tee -a "$FAILED_TXT"
    fi
    failed=$((failed + 1))
  fi
done

echo "[DONE] logs saved to: $LOG_DIR"
if [[ "$LOG_FILE_MODE" == "two" ]]; then
  echo "[DONE] server log: $SERVER_LOG_FILE"
  echo "[DONE] bench log: $BENCH_LOG_FILE"
else
  echo "[DONE] summary file: $SUMMARY_CSV"
fi

if [[ "$failed" -gt 0 ]]; then
  if [[ "$LOG_FILE_MODE" == "two" ]]; then
    echo "[DONE] failed cases: $failed (${FAILED_CASES[*]})"
  else
    echo "[DONE] failed cases: $failed (details: $FAILED_TXT)"
  fi
  exit 2
fi









