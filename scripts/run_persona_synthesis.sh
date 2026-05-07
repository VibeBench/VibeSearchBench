#!/usr/bin/env bash
# ============================================================================
# VIBEResearch — Persona-Only Synthesis
#
# Synthesize user_persona only; keep existing initial_query and final_query.
#
# Usage:
#   bash scripts/run_persona_synthesis.sh
#   MODEL_NAME=qwen3-30b VLLM_URL=http://host/v1 bash scripts/run_persona_synthesis.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"

export no_proxy=""
export NO_PROXY="$no_proxy"

# ---------- Config ----------
MODEL_NAME="${MODEL_NAME:-doubao-seed-2-0-pro-260215}"
VLLM_URL="${VLLM_URL:-https://example.com/openai/doubao}"
API_KEY="${API_KEY:-your_api_key_here}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-128000}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-40}"

TASK_DIR="${TASK_DIR:-$PROJECT_DIR/tasks/0427_sample_100}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/tasks/vrb_sample_v_0_1}"

# ---------- Run ----------
echo ">>> Persona-Only Synthesis"
echo "    Task dir:   $TASK_DIR"
echo "    Output dir: $OUTPUT_DIR"
echo "    Model:      $MODEL_NAME"
echo "    URL:        $VLLM_URL"
echo ""

$PYTHON viberesearch_query_synthesis/synthesize.py \
  --task-dir "$TASK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --base-url "$VLLM_URL" \
  --model "$MODEL_NAME" \
  --api-key "$API_KEY" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --persona-only
