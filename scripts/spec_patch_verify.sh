#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Required paths
MODEL_PATH="${MODEL_PATH:-/mnt/ky2307909/siyuan.tong/Qwen3-8B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/mnt/ky2307909/siyuan.tong/Qwen3-0-6B}"
DATASET_PATH="${DATASET_PATH:-/mnt/ky2307909/siyuan.tong/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}"

# Benchmark settings
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8B}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-16}"

# Keep one K list here.
K_LIST_STR="${K_LIST_STR:-0 2 4 8}"

# Ablation suites (space-separated)
# A: FP on + MQA on
# B: FP on + MQA off
# C: FP off + MQA on
# D: FP off + MQA off
# NO_SPEC: no speculative decoding baseline
# Also supports: legacy / patched
SUITES_STR="${SUITES_STR:-A_FP_ON_MQA_ON B_FP_ON_MQA_OFF C_FP_OFF_MQA_ON D_FP_OFF_MQA_OFF NO_SPEC}"

READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-0}"
FAIL_FAST="${FAIL_FAST:-1}"

ROOT_DIR="${ROOT_DIR:-${WORKSPACE_DIR}}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${ROOT_DIR}/logs/spec_patch_verify_${RUN_ID}"
mkdir -p "${LOG_ROOT}"
SUMMARY_CSV="${LOG_ROOT}/summary.csv"

# Force runtime to prioritize local patched vllm-ascend code.
export PYTHONPATH="${WORKSPACE_DIR}/vllm-ascend:${PYTHONPATH:-}"

# Fixed env for V0 speculative decode on Ascend
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
export VLLM_ASCEND_ADAPTIVE_K_ENABLE="${VLLM_ASCEND_ADAPTIVE_K_ENABLE:-0}"
unset CUDA_VISIBLE_DEVICES || true

# Logging controls
export VLLM_ASCEND_MS_STATS_INTERVAL="${VLLM_ASCEND_MS_STATS_INTERVAL:-1}"
export VLLM_ASCEND_BASE_STAGE_LOG_ENABLE="${VLLM_ASCEND_BASE_STAGE_LOG_ENABLE:-1}"
export VLLM_ASCEND_BASE_STAGE_LOG_INTERVAL="${VLLM_ASCEND_BASE_STAGE_LOG_INTERVAL:-50}"
export VLLM_ASCEND_RUNNER_STAGE_LOG_ENABLE="${VLLM_ASCEND_RUNNER_STAGE_LOG_ENABLE:-1}"
export VLLM_ASCEND_RUNNER_STAGE_LOG_INTERVAL="${VLLM_ASCEND_RUNNER_STAGE_LOG_INTERVAL:-50}"

append_space_split_args() {
  local arg_string="$1"
  local -n target_arr=$2
  [[ -z "${arg_string}" ]] && return 0
  local extra=()
  # shellcheck disable=SC2206
  extra=(${arg_string})
  target_arr+=("${extra[@]}")
}

parse_k_list() {
  read -r -a K_LIST <<< "${K_LIST_STR}"
  if [[ ${#K_LIST[@]} -eq 0 ]]; then
    echo "[ERROR] empty K_LIST_STR"
    exit 1
  fi
  local k
  for k in "${K_LIST[@]}"; do
    if ! [[ "${k}" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] invalid k in K_LIST_STR: ${k}"
      exit 1
    fi
  done
}

require_paths() {
  if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
    exit 1
  fi

  if [[ -d "${MODEL_PATH}" ]]; then
    if [[ ! -f "${MODEL_PATH}/config.json" ]] && [[ ! -f "${MODEL_PATH}/params.json" ]]; then
      echo "[ERROR] MODEL_PATH dir has no config.json or params.json: ${MODEL_PATH}"
      exit 1
    fi
  fi

  if [[ ! -e "${DRAFT_MODEL_PATH}" ]]; then
    echo "[ERROR] DRAFT_MODEL_PATH not found: ${DRAFT_MODEL_PATH}"
    exit 1
  fi

  if [[ -d "${DRAFT_MODEL_PATH}" ]]; then
    if [[ ! -f "${DRAFT_MODEL_PATH}/config.json" ]] && [[ ! -f "${DRAFT_MODEL_PATH}/params.json" ]]; then
      echo "[ERROR] DRAFT_MODEL_PATH dir has no config.json or params.json: ${DRAFT_MODEL_PATH}"
      exit 1
    fi
  fi

  if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "[ERROR] DATASET_PATH not found: ${DATASET_PATH}"
    exit 1
  fi
}

runtime_probe() {
  local suite="$1"
  local probe_log="${LOG_ROOT}/probe_${suite}.log"

  {
    echo "[PROBE] date=$(date '+%F %T')"
    echo "[PROBE] WORKSPACE_DIR=${WORKSPACE_DIR}"
    echo "[PROBE] ROOT_DIR=${ROOT_DIR}"
    echo "[PROBE] LOG_ROOT=${LOG_ROOT}"
    echo "[PROBE] PYTHONPATH=${PYTHONPATH}"
    echo "[PROBE] VLLM_ASCEND_SPEC_GPU_MULTI_STEP_BACKENDS=${VLLM_ASCEND_SPEC_GPU_MULTI_STEP_BACKENDS:-}"
    echo "[PROBE] VLLM_ASCEND_SPEC_MQA_BACKENDS=${VLLM_ASCEND_SPEC_MQA_BACKENDS:-}"
    echo "[PROBE] VLLM_ASCEND_BASE_STAGE_LOG_ENABLE=${VLLM_ASCEND_BASE_STAGE_LOG_ENABLE:-}"
    echo "[PROBE] VLLM_ASCEND_BASE_STAGE_LOG_INTERVAL=${VLLM_ASCEND_BASE_STAGE_LOG_INTERVAL:-}"
    echo "[PROBE] VLLM_ASCEND_RUNNER_STAGE_LOG_ENABLE=${VLLM_ASCEND_RUNNER_STAGE_LOG_ENABLE:-}"
    echo "[PROBE] VLLM_ASCEND_RUNNER_STAGE_LOG_INTERVAL=${VLLM_ASCEND_RUNNER_STAGE_LOG_INTERVAL:-}"

    python - <<'PY'
import inspect

def src(mod):
    return inspect.getsourcefile(mod) or inspect.getfile(mod)

import vllm_ascend
import vllm_ascend.worker.worker as worker
import vllm_ascend.worker.model_runner as runner
import vllm_ascend.worker.draft_model_runner as draft
import vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker as patch_spec
import vllm_ascend.patch.worker.patch_common.patch_multi_step_worker as patch_ms

p0 = src(vllm_ascend)
p1 = src(worker)
p2 = src(runner)
p3 = src(draft)
p4 = src(patch_spec)
p5 = src(patch_ms)

print(f"[PROBE] vllm_ascend={p0}")
print(f"[PROBE] worker={p1}")
print(f"[PROBE] model_runner={p2}")
print(f"[PROBE] draft_model_runner={p3}")
print(f"[PROBE] patch_spec_decode_worker={p4}")
print(f"[PROBE] patch_multi_step_worker={p5}")

checks = [
    ("worker_has_base_stage_log", p1, "BaseWorker stage times"),
    ("runner_has_base_stage_log", p2, "BaseRunner stage times"),
    ("draft_has_fastpath_log", p3, "DraftGPUFastPath"),
]
for name, path, token in checks:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        print(f"[PROBE] {name}={token in txt}")
    except Exception as exc:
        print(f"[PROBE] {name}=ERROR:{exc}")
PY
  } | tee "${probe_log}"
}

wait_server_ready() {
  local pid="$1"
  local timeout_s="$2"
  local i=0
  while (( i < timeout_s )); do
    if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[INFO] server ready after ${i}s"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] server process exited before ready"
      return 2
    fi
    sleep 1
    ((i+=1))
  done
  echo "[ERROR] server not ready in ${timeout_s}s"
  return 1
}

stop_server() {
  local pid="${1:-}"
  set +e
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  pkill -TERM -f "vllm serve|api_server.py|engine.py|multiprocessing.engine" 2>/dev/null || true
  sleep 1
  pkill -KILL -f "vllm serve|api_server.py|engine.py|multiprocessing.engine" 2>/dev/null || true
  set -e
}

resolve_suite_backends() {
  local suite="$1"
  case "${suite}" in
    A_FP_ON_MQA_ON|patched)
      echo "FLASH_ATTN,ASCEND|FLASH_ATTN,ASCEND"
      ;;
    B_FP_ON_MQA_OFF)
      echo "FLASH_ATTN,ASCEND|FLASH_ATTN"
      ;;
    C_FP_OFF_MQA_ON)
      echo "FLASH_ATTN|FLASH_ATTN,ASCEND"
      ;;
    D_FP_OFF_MQA_OFF|legacy)
      echo "FLASH_ATTN|FLASH_ATTN"
      ;;
    NO_SPEC)
      echo "FLASH_ATTN,ASCEND|FLASH_ATTN,ASCEND"
      ;;
    *)
      echo "[ERROR] unknown suite: ${suite}" >&2
      return 1
      ;;
  esac
}

extract_metric_num() {
  local file="$1"
  local pattern="$2"
  local line
  line="$(grep -E -m1 "${pattern}" "${file}" 2>/dev/null || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return 0
  fi
  echo "${line}" | sed -E 's/.*: *([0-9]+(\.[0-9]+)?).*/\1/'
}

extract_last_num() {
  local file="$1"
  local pattern="$2"
  local sed_expr="$3"
  local line
  line="$(grep -E "${pattern}" "${file}" 2>/dev/null | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return 0
  fi
  echo "${line}" | sed -E "${sed_expr}"
}

append_summary_row() {
  local suite="$1"
  local k="$2"
  local status="$3"
  local serve_log="$4"
  local bench_log="$5"
  local key_log="$6"

  local throughput tpot itl mqa_mode supports_ms fastpath_hit scoring_ms verify_ms base_spmd_ms runner_total_ms

  throughput="$(extract_metric_num "${bench_log}" 'Output token throughput')"
  tpot="$(extract_metric_num "${bench_log}" 'Mean TPOT')"
  itl="$(extract_metric_num "${bench_log}" 'Mean ITL')"

  if grep -q 'Use MQA scorer for scoring proposals' "${key_log}" 2>/dev/null; then
    mqa_mode="MQA"
  elif grep -q 'Use batch expansion for scoring proposals' "${key_log}" 2>/dev/null; then
    mqa_mode="BATCH_EXPANSION"
  else
    mqa_mode="UNKNOWN"
  fi

  supports_ms="$(extract_last_num "${key_log}" 'supports_gpu_multi_step=' 's/.*supports_gpu_multi_step=([^ ]+).*/\1/')"
  fastpath_hit="$(extract_last_num "${key_log}" 'DraftGPUFastPath hit_rate' 's/.*hit_rate=([0-9.]+).*/\1/')"

  scoring_ms="$(extract_last_num "${key_log}" 'SpecDecodeWorker stage times' 's/.*scoring_time_ms=([0-9.]+).*/\1/')"
  verify_ms="$(extract_last_num "${key_log}" 'SpecDecodeWorker stage times' 's/.*verification_time_ms=([0-9.]+).*/\1/')"
  base_spmd_ms="$(extract_last_num "${key_log}" 'BaseWorker stage times' 's/.*avg_total_spmd_ms=([0-9.]+).*/\1/')"
  runner_total_ms="$(extract_last_num "${key_log}" 'BaseRunner stage times' 's/.*avg_execute_total_ms=([0-9.]+).*/\1/')"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${suite}" "${k}" "${status}" "${throughput}" "${tpot}" "${itl}" \
    "${mqa_mode}" "${supports_ms}" "${fastpath_hit}" "${scoring_ms}" \
    "${verify_ms}" "${base_spmd_ms}|${runner_total_ms}" >> "${SUMMARY_CSV}"
}

run_case() {
  local suite="$1"
  local k="$2"
  local tag="${suite}_k${k}"
  local serve_log="${LOG_ROOT}/server_${tag}.log"
  local bench_log="${LOG_ROOT}/bench_${tag}.log"
  local key_log="${LOG_ROOT}/key_${tag}.log"
  local meta_log="${LOG_ROOT}/meta_${tag}.log"

  echo "========== ${tag} START $(date '+%F %T') =========="

  stop_server ""

  local -a serve_cmd=(
    vllm serve "${MODEL_PATH}"
    --gpu-memory-utilization 0.90
    --max-model-len 8192
    --served-model-name "${SERVED_MODEL_NAME}"
    --trust-remote-code
    --port "${PORT}"
    --host "${HOST}"
    --enforce-eager
  )

  if [[ "${suite}" != "NO_SPEC" ]]; then
    local spec_cfg
    printf -v spec_cfg '{"method":"draft_model","model":"%s","num_speculative_tokens":%d}' \
      "${DRAFT_MODEL_PATH}" "${k}"
    serve_cmd+=(--speculative-config "${spec_cfg}")
  fi

  append_space_split_args "${SERVE_EXTRA_ARGS:-}" serve_cmd

  {
    echo "[META] date=$(date '+%F %T')"
    echo "[META] suite=${suite}"
    echo "[META] k=${k}"
    echo "[META] MODEL_PATH=${MODEL_PATH}"
    echo "[META] DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH}"
    echo "[META] DATASET_PATH=${DATASET_PATH}"
    echo "[META] HOST=${HOST} PORT=${PORT}"
    echo "[META] NUM_PROMPTS=${NUM_PROMPTS} REQUEST_RATE=${REQUEST_RATE}"
    echo "[META] VLLM_ASCEND_SPEC_GPU_MULTI_STEP_BACKENDS=${VLLM_ASCEND_SPEC_GPU_MULTI_STEP_BACKENDS:-}"
    echo "[META] VLLM_ASCEND_SPEC_MQA_BACKENDS=${VLLM_ASCEND_SPEC_MQA_BACKENDS:-}"
    echo "[META] VLLM_ASCEND_BASE_STAGE_LOG_ENABLE=${VLLM_ASCEND_BASE_STAGE_LOG_ENABLE:-}"
    echo "[META] VLLM_ASCEND_BASE_STAGE_LOG_INTERVAL=${VLLM_ASCEND_BASE_STAGE_LOG_INTERVAL:-}"
    echo "[META] VLLM_ASCEND_RUNNER_STAGE_LOG_ENABLE=${VLLM_ASCEND_RUNNER_STAGE_LOG_ENABLE:-}"
    echo "[META] VLLM_ASCEND_RUNNER_STAGE_LOG_INTERVAL=${VLLM_ASCEND_RUNNER_STAGE_LOG_INTERVAL:-}"
    printf '[META] serve_cmd='; printf '%q ' "${serve_cmd[@]}"; echo
  } > "${meta_log}"

  "${serve_cmd[@]}" >"${serve_log}" 2>&1 &
  local server_pid=$!

  if ! wait_server_ready "${server_pid}" "${READY_TIMEOUT_S}"; then
    echo "[ERROR] server failed: ${tag}"
    tail -n 200 "${serve_log}" || true
    stop_server "${server_pid}"
    grep -E \
"DraftGPUFastPath|MultiStep probe|supports_gpu_multi_step|supports_spec_gpu_multi_step|_attn_metadata_supports_advance_step|Draft runner_cls|MQA scorer backend capability|Disabling MQA scorer|Use MQA scorer|Use batch expansion|SpecDecodeWorker stage times|BaseWorker stage times|BaseRunner stage times|Speculative metrics|P1P2|ALIGN P1/P2|Traceback|ERROR|ValueError|RuntimeError" \
      "${serve_log}" > "${key_log}" || true
    append_summary_row "${suite}" "${k}" "SERVER_FAIL" "${serve_log}" "${bench_log}" "${key_log}"
    if [[ "${FAIL_FAST}" == "1" ]]; then
      exit 1
    fi
    return 0
  fi

  local -a bench_cmd=(
    vllm bench serve
    --model "${SERVED_MODEL_NAME}"
    --tokenizer "${MODEL_PATH}"
    --base-url "http://${HOST}:${PORT}"
    --endpoint-type vllm
    --dataset-name sharegpt
    --dataset-path "${DATASET_PATH}"
    --num-prompts "${NUM_PROMPTS}"
    --request-rate "${REQUEST_RATE}"
    --trust-remote-code
  )
  append_space_split_args "${BENCH_EXTRA_ARGS:-}" bench_cmd
  {
    printf '[META] bench_cmd='; printf '%q ' "${bench_cmd[@]}"; echo
  } >> "${meta_log}"

  local bench_rc=0
  set +e
  if [[ "${BENCH_TIMEOUT_S}" =~ ^[0-9]+$ ]] && (( BENCH_TIMEOUT_S > 0 )) && command -v timeout >/dev/null 2>&1; then
    timeout "${BENCH_TIMEOUT_S}s" "${bench_cmd[@]}" 2>&1 | tee "${bench_log}"
    bench_rc=${PIPESTATUS[0]}
  else
    "${bench_cmd[@]}" 2>&1 | tee "${bench_log}"
    bench_rc=${PIPESTATUS[0]}
  fi
  set -e

  stop_server "${server_pid}"

  grep -E \
"DraftGPUFastPath|MultiStep probe|supports_gpu_multi_step|supports_spec_gpu_multi_step|_attn_metadata_supports_advance_step|Draft runner_cls|MQA scorer backend capability|Disabling MQA scorer|Use MQA scorer|Use batch expansion|SpecDecodeWorker stage times|BaseWorker stage times|BaseRunner stage times|Speculative metrics|Output token throughput|Mean TPOT|Mean ITL|P1P2|ALIGN P1/P2|Traceback|ERROR|ValueError|RuntimeError" \
    "${serve_log}" "${bench_log}" > "${key_log}" || true

  if [[ ${bench_rc} -eq 0 ]]; then
    append_summary_row "${suite}" "${k}" "OK" "${serve_log}" "${bench_log}" "${key_log}"
  else
    append_summary_row "${suite}" "${k}" "BENCH_FAIL(${bench_rc})" "${serve_log}" "${bench_log}" "${key_log}"
  fi

  echo "[INFO] saved:"
  echo "  serve: ${serve_log}"
  echo "  bench: ${bench_log}"
  echo "  key:   ${key_log}"
  echo "  meta:  ${meta_log}"
  echo "========== ${tag} END $(date '+%F %T') =========="

  if [[ ${bench_rc} -ne 0 && "${FAIL_FAST}" == "1" ]]; then
    echo "[ERROR] bench failed for ${tag}, rc=${bench_rc}"
    exit "${bench_rc}"
  fi
}

run_suite() {
  local suite="$1"
  local pair
  pair="$(resolve_suite_backends "${suite}")"
  local gpu_backends="${pair%%|*}"
  local mqa_backends="${pair##*|}"

  echo "############################"
  echo "[SUITE] ${suite}"
  echo "VLLM_ASCEND_SPEC_GPU_MULTI_STEP_BACKENDS=${gpu_backends}"
  echo "VLLM_ASCEND_SPEC_MQA_BACKENDS=${mqa_backends}"
  echo "K_LIST=${K_LIST[*]}"
  echo "############################"

  export VLLM_ASCEND_SPEC_GPU_MULTI_STEP_BACKENDS="${gpu_backends}"
  export VLLM_ASCEND_SPEC_MQA_BACKENDS="${mqa_backends}"

  runtime_probe "${suite}"

  local k
  for k in "${K_LIST[@]}"; do
    if [[ "${suite}" == "NO_SPEC" && "${k}" != "0" ]]; then
      echo "[INFO] skip ${suite} with k=${k} (NO_SPEC only runs k=0)"
      continue
    fi
    run_case "${suite}" "${k}"
  done
}

main() {
  parse_k_list
  require_paths

  echo 'suite,k,status,throughput_tps,mean_tpot_ms,mean_itl_ms,mqa_mode,supports_gpu_multi_step,fastpath_hit_rate,scoring_time_ms,verification_time_ms,base_worker_total_ms|base_runner_total_ms' > "${SUMMARY_CSV}"

  echo "[INFO] LOG_ROOT=${LOG_ROOT}"
  echo "[INFO] SUMMARY_CSV=${SUMMARY_CSV}"
  echo "[INFO] SUITES=${SUITES_STR}"
  echo "[INFO] K_LIST=${K_LIST_STR}"

  local suite
  for suite in ${SUITES_STR}; do
    run_suite "${suite}"
  done

  echo "[DONE] all logs: ${LOG_ROOT}"
  echo "[DONE] summary csv: ${SUMMARY_CSV}"
}

main "$@"