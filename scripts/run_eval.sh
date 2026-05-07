#!/usr/bin/env bash
# ============================================================================
# VIBEResearch — Evaluation Only (on existing trajectories)
#
# Usage:
#   TRAJS_DIR=results/trajs/glm-5.1_custom_serper bash scripts/run_eval.sh
#
#   # With custom task folder and eval output dir:
#   TRAJS_DIR=results/trajs/xxx DATA_PATH=my_tasks/ EVAL_DIR=my_eval/ bash scripts/run_eval.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"

export no_proxy=""
export NO_PROXY="$no_proxy"

# ---------- Config ----------
TRAJS_DIR="${TRAJS_DIR:?'Set TRAJS_DIR to the trajectory directory, e.g. results/trajs/glm-5.1_custom_serper'}"
DATA_PATH="${DATA_PATH:-$PROJECT_DIR/tasks/vrb_final}"
EVAL_DIR="${EVAL_DIR:-}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"

GRADER_TYPE="${GRADER_TYPE:-openai}"
GRADER_BASE_URL="${GRADER_BASE_URL:-http://localhost:30000/v1}"
GRADER_MODEL="${GRADER_MODEL:-Qwen3.5-397B-A17B-FP8}"
GRADER_API_KEY="${GRADER_API_KEY:-EMPTY}"
GRADER_THREADS="${GRADER_THREADS:-20}"

# Legacy Gemini settings (used when GRADER_TYPE=gemini)
GEMINI_API_KEY="${GEMINI_API_KEY:-your_api_key_here}"
GEMINI_API_URL="${GEMINI_API_URL:-https://example.com/gemini}"

EXTRA=()
[[ -n "${EVAL_DIR:-}" ]] && EXTRA+=(--eval-dir "$EVAL_DIR")

echo ">>> VIBEResearch Evaluation (trajs=$TRAJS_DIR data=$DATA_PATH grader=$GRADER_TYPE model=$GRADER_MODEL)"
if [[ "$GRADER_TYPE" == "gemini" ]]; then
  $PYTHON run.py \
    --eval-only \
    --trajs-dir "$TRAJS_DIR" \
    --data-path "$DATA_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --grader-type gemini \
    --grader-api-url "$GEMINI_API_URL" \
    --grader-api-key "$GEMINI_API_KEY" \
    --grader-threads "$GRADER_THREADS" \
    "${EXTRA[@]}" \
    2>&1
else
  $PYTHON run.py \
    --eval-only \
    --trajs-dir "$TRAJS_DIR" \
    --data-path "$DATA_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --grader-type "$GRADER_TYPE" \
    --grader-base-url "$GRADER_BASE_URL" \
    --grader-model "$GRADER_MODEL" \
    --grader-api-key "$GRADER_API_KEY" \
    --grader-threads "$GRADER_THREADS" \
    "${EXTRA[@]}" \
    2>&1
fi
