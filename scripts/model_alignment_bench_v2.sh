#!/usr/bin/env bash
set -euo pipefail

# v2 wrapper for model-alignment experiments.
# Adds new calibration knobs introduced in draft_model_runner.py and then
# delegates matrix execution to model_alignment_bench.sh.

export VLLM_ASCEND_DRAFT_ALIGN_MODE="${VLLM_ASCEND_DRAFT_ALIGN_MODE:-temperature}"
export VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE="${VLLM_ASCEND_DRAFT_ALIGN_TEMPERATURE:-1.0}"
export VLLM_ASCEND_DRAFT_ALIGN_VOCAB_BIAS_PATH="${VLLM_ASCEND_DRAFT_ALIGN_VOCAB_BIAS_PATH:-}"

# Existing script still controls A0/A1/A2 matrix and scale/bias envs.
exec bash "$(dirname "$0")/model_alignment_bench.sh" "$@"
