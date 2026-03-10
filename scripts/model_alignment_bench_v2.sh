#!/usr/bin/env bash
set -euo pipefail

# v2 wrapper for model-alignment experiments.
# Adds new calibration knobs introduced in draft_model_runner.py and then
# delegates matrix execution to model_alignment_bench.sh.
#
# Optional human-readable controls:
#   EXPERIMENT_MODE: legacy|basedraft|align_only|adaptive_only|joint|all
#                    (aliases: base_draft|align|logits|adaptive|*|matrix)
#   K_LIST: comma-separated fixed-k list for non-adaptive modes, e.g. 2,4,8
#   ADAPTIVE_INIT_K_LIST: comma-separated init-k list for adaptive modes
#   ADAPTIVE_K_MIN / ADAPTIVE_K_MAX
#   ADAPTIVE_ENABLE_UTILITY: 0/1
#   LOG_ROOT: root dir for multi-run logs (defaults to LOG_DIR or timestamp dir)
#
# Backward compatibility:
#   EXPERIMENT_MODE=legacy (default) keeps old behavior and relies on
#   RUN_A0/RUN_A1/RUN_A2 + FIXED_K_LIST_STR.

export VLLM_ASCEND_DRAFT_ALIGN_MODE="${VLLM_ASCEND_DRAFT_ALIGN_MODE:-temperature}"
export VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE="${VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE:-1.0}"
export VLLM_ASCEND_DRAFT_ALIGN_VOCAB_BIAS_PATH="${VLLM_ASCEND_DRAFT_ALIGN_VOCAB_BIAS_PATH:-}"

# Optional phase-4 explicit linkage defaults (off by default).
export VLLM_ASCEND_ADAPTIVE_ALIGN_GATE_ENABLE="${VLLM_ASCEND_ADAPTIVE_ALIGN_GATE_ENABLE:-0}"
export VLLM_ASCEND_ADAPTIVE_ALIGN_LOW_TH="${VLLM_ASCEND_ADAPTIVE_ALIGN_LOW_TH:-0.45}"
export VLLM_ASCEND_ADAPTIVE_ALIGN_HIGH_TH="${VLLM_ASCEND_ADAPTIVE_ALIGN_HIGH_TH:-0.70}"
export VLLM_ASCEND_ADAPTIVE_ALIGN_CAP_LOW="${VLLM_ASCEND_ADAPTIVE_ALIGN_CAP_LOW:-3}"
export VLLM_ASCEND_ADAPTIVE_ALIGN_CAP_HIGH="${VLLM_ASCEND_ADAPTIVE_ALIGN_CAP_HIGH:-8}"
export VLLM_ASCEND_ADAPTIVE_ALIGN_W_POS2="${VLLM_ASCEND_ADAPTIVE_ALIGN_W_POS2:-0.45}"
export VLLM_ASCEND_ADAPTIVE_ALIGN_W_NOWASTE="${VLLM_ASCEND_ADAPTIVE_ALIGN_W_NOWASTE:-0.45}"
export VLLM_ASCEND_ADAPTIVE_ALIGN_W_DELTA_HS="${VLLM_ASCEND_ADAPTIVE_ALIGN_W_DELTA_HS:-0.10}"

# Keep only two output files by default: server.log and bench.log.
export LOG_FILE_MODE="${LOG_FILE_MODE:-two}"

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/model_alignment_bench.sh"

EXPERIMENT_MODE="${EXPERIMENT_MODE:-legacy}"
K_LIST="${K_LIST:-${FIXED_K_LIST_STR:-2,4,8}}"
ADAPTIVE_INIT_K_LIST="${ADAPTIVE_INIT_K_LIST:-${K_LIST}}"
ADAPTIVE_K_MIN="${ADAPTIVE_K_MIN:-${VLLM_ASCEND_ADAPTIVE_K_MIN:-2}}"
ADAPTIVE_K_MAX="${ADAPTIVE_K_MAX:-${VLLM_ASCEND_ADAPTIVE_K_MAX:-8}}"
ADAPTIVE_ENABLE_UTILITY="${ADAPTIVE_ENABLE_UTILITY:-${VLLM_ASCEND_ADAPTIVE_ENABLE_UTILITY:-1}}"
LOG_ROOT="${LOG_ROOT:-${LOG_DIR:-./logs/model_alignment_v2_$(date +%Y%m%d_%H%M%S)}}"

split_csv() {
  local input="$1"
  local -n out_ref=$2
  out_ref=()
  [[ -z "$input" ]] && return 0
  IFS=',' read -r -a out_ref <<<"$input"
}

normalize_mode() {
  local mode="$1"
  case "$mode" in
    base_draft|base) echo "basedraft" ;;
    align|logits) echo "align_only" ;;
    adaptive) echo "adaptive_only" ;;
    '*'|matrix) echo "all" ;;
    *) echo "$mode" ;;
  esac
}

configure_mode_flags() {
  local mode="$1"
  export RUN_A0=0
  export RUN_A1=0
  export RUN_A2=0
  case "$mode" in
    basedraft) export RUN_A0=1 ;;
    adaptive_only) export RUN_A1=1 ;;
    align_only|joint) export RUN_A2=1 ;;
    *)
      echo "[ERROR] unsupported mode: ${mode}" >&2
      return 1
      ;;
  esac
}

run_mode() {
  local mode="$1"
  local init_k="$2"
  shift 2

  local run_tag="$mode"

  configure_mode_flags "$mode"

  if [[ "$mode" == "adaptive_only" || "$mode" == "joint" ]]; then
    export VLLM_ASCEND_ADAPTIVE_K_ENABLE=1
    export VLLM_ASCEND_ADAPTIVE_ENABLE_UTILITY="$ADAPTIVE_ENABLE_UTILITY"
    export VLLM_ASCEND_ADAPTIVE_K_INIT="$init_k"
    export VLLM_ASCEND_ADAPTIVE_K_MIN="$ADAPTIVE_K_MIN"
    export VLLM_ASCEND_ADAPTIVE_K_MAX="$ADAPTIVE_K_MAX"

    # For adaptive runs, serve with upper bound k to allow runtime up/down.
    export FIXED_K_LIST_STR="$ADAPTIVE_K_MAX"
    run_tag="${mode}_initk${init_k}"
  else
    export VLLM_ASCEND_ADAPTIVE_K_ENABLE=0
    export VLLM_ASCEND_ADAPTIVE_ENABLE_UTILITY=0
    unset VLLM_ASCEND_ADAPTIVE_K_INIT || true
    export VLLM_ASCEND_ADAPTIVE_K_MIN="$ADAPTIVE_K_MIN"
    export VLLM_ASCEND_ADAPTIVE_K_MAX="$ADAPTIVE_K_MAX"

    export FIXED_K_LIST_STR="$K_LIST"
  fi

  if [[ "$LOG_FILE_MODE" == "two" ]]; then
    export LOG_DIR="$LOG_ROOT"
    if (( RUN_INDEX > 0 )); then
      export LOG_FILE_APPEND=1
    else
      export LOG_FILE_APPEND=0
    fi
  else
    export LOG_DIR="${LOG_ROOT}/${run_tag}"
    export LOG_FILE_APPEND=0
  fi

  echo "[v2] mode=${mode} fixed_k=${FIXED_K_LIST_STR} init_k=${VLLM_ASCEND_ADAPTIVE_K_INIT:-0} k_min=${VLLM_ASCEND_ADAPTIVE_K_MIN:-0} k_max=${VLLM_ASCEND_ADAPTIVE_K_MAX:-0}"
  bash "$BASE_SCRIPT" "$@"
  RUN_INDEX=$((RUN_INDEX + 1))
}

if [[ "$EXPERIMENT_MODE" == "legacy" ]]; then
  # Existing script controls A0/A1/A2 matrix and scale/bias envs.
  exec bash "$BASE_SCRIPT" "$@"
fi

if [[ "$ADAPTIVE_K_MIN" =~ ^[0-9]+$ ]] && [[ "$ADAPTIVE_K_MAX" =~ ^[0-9]+$ ]] && (( ADAPTIVE_K_MIN > ADAPTIVE_K_MAX )); then
  tmp="$ADAPTIVE_K_MIN"
  ADAPTIVE_K_MIN="$ADAPTIVE_K_MAX"
  ADAPTIVE_K_MAX="$tmp"
fi

MODE_RAW="$(normalize_mode "$EXPERIMENT_MODE")"
declare -a MODE_LIST=()
declare -a INIT_K_LIST=()

if [[ "$MODE_RAW" == "all" ]]; then
  MODE_LIST=(basedraft align_only adaptive_only joint)
else
  MODE_LIST=("$MODE_RAW")
fi

split_csv "$ADAPTIVE_INIT_K_LIST" INIT_K_LIST

RUN_INDEX=0

for mode in "${MODE_LIST[@]}"; do
  case "$mode" in
    basedraft|align_only)
      run_mode "$mode" "" "$@"
      ;;
    adaptive_only|joint)
      if [[ ${#INIT_K_LIST[@]} -eq 0 ]]; then
        echo "[ERROR] ADAPTIVE_INIT_K_LIST is empty for mode=${mode}" >&2
        exit 1
      fi
      for init_k in "${INIT_K_LIST[@]}"; do
        [[ -z "$init_k" ]] && continue
        run_mode "$mode" "$init_k" "$@"
      done
      ;;
    *)
      echo "[ERROR] unknown EXPERIMENT_MODE: ${mode}" >&2
      exit 1
      ;;
  esac
done




