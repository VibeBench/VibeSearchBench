#!/usr/bin/env bash
# ============================================================================
# VIBEResearch — OpenClaw Evaluation (Inference + Evaluation)
#
# Evaluates OpenClaw through the viberesearch benchmark.
# Supports all three modes: direct / staged / simulated.
#
# Usage:
#   bash scripts/run_openclaw.sh
#   MODE=simulated bash scripts/run_openclaw.sh
#   DATA_PATH=tasks/debug MODE=direct bash scripts/run_openclaw.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"

export no_proxy=""
export NO_PROXY="$no_proxy"

# ---------- OpenClaw config ----------
GATEWAY_PORT="${GATEWAY_PORT:-18789}"
SOURCE_DIR="${SOURCE_DIR:-./openclaw_backup}"
OPENCLAW_RESULTS_DIR="${OPENCLAW_RESULTS_DIR:-}"
IDLE_THRESHOLD="${IDLE_THRESHOLD:-90}"
MAX_NUDGE="${MAX_NUDGE:-3}"
OPENCLAW_MODEL="${OPENCLAW_MODEL:-}"

# ---------- Task data ----------
DATA_PATH="${DATA_PATH:-$PROJECT_DIR/tasks/0423_debug_10}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"

# ---------- Execution mode: direct / staged / simulated ----------
MODE="${MODE:-simulated}"
MAX_USER_TURNS="${MAX_USER_TURNS:-50}"

# ---------- User simulator LLM (for simulated mode) ----------
USER_MODEL="${USER_MODEL:-doubao-seed-2-0-pro-260215}"
USER_MODEL_URL="${USER_MODEL_URL:-https://example.com/openai/doubao}"
USER_MODEL_API_KEY="${USER_MODEL_API_KEY:-your_api_key_here}"

# ---------- Grader ----------
GEMINI_API_KEY="${GEMINI_API_KEY:-your_api_key_here}"
GEMINI_API_URL="${GEMINI_API_URL:-https://example.com/gemini}"
GRADER_TYPE="${GRADER_TYPE:-gemini}"
GRADER_THREADS="${GRADER_THREADS:-16}"

# ---------- Log ----------
LOG_DIR="$PROJECT_DIR/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/openclaw_$(date +%Y%m%d_%H%M%S).log}"

# ---------- Build command ----------
EXTRA=()

# OpenClaw-specific args
EXTRA+=(--agent-type openclaw)
EXTRA+=(--gateway-port "$GATEWAY_PORT")
EXTRA+=(--source-dir "$SOURCE_DIR")
[[ -n "${OPENCLAW_RESULTS_DIR:-}" ]] && EXTRA+=(--openclaw-results-dir "$OPENCLAW_RESULTS_DIR")
EXTRA+=(--idle-threshold "$IDLE_THRESHOLD")
EXTRA+=(--max-nudge "$MAX_NUDGE")
[[ -n "${OPENCLAW_MODEL:-}" ]] && EXTRA+=(--openclaw-model "$OPENCLAW_MODEL")

# Mode
EXTRA+=(--mode "$MODE")

# Simulated mode needs a user simulator LLM
if [[ "$MODE" == "simulated" ]]; then
  EXTRA+=(--max-user-turns "$MAX_USER_TURNS")
  EXTRA+=(--vllm-server-url "$USER_MODEL_URL")
  EXTRA+=(--model "$USER_MODEL")
  EXTRA+=(--api-key "$USER_MODEL_API_KEY")
  [[ -n "${USER_MODEL:-}" ]] && EXTRA+=(--user-model "$USER_MODEL")
  [[ -n "${USER_MODEL_URL:-}" ]] && EXTRA+=(--user-model-url "$USER_MODEL_URL")
  [[ -n "${USER_MODEL_API_KEY:-}" ]] && EXTRA+=(--user-model-api-key "$USER_MODEL_API_KEY")
fi

DATASET_NAME="$(basename "$DATA_PATH")"
echo ">>> VIBEResearch OpenClaw Evaluation (dataset=$DATASET_NAME mode=$MODE)"
echo ">>> Gateway: port=$GATEWAY_PORT source=$SOURCE_DIR"
echo ">>> Log: $LOG_FILE"

$PYTHON run.py \
  --data-path "$DATA_PATH" \
  --num-samples "$NUM_SAMPLES" \
  --grader-type "$GRADER_TYPE" \
  --grader-api-url "$GEMINI_API_URL" \
  --grader-api-key "$GEMINI_API_KEY" \
  --grader-threads "$GRADER_THREADS" \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG_FILE"
