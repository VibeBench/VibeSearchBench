#!/usr/bin/env bash
# ============================================================================
# VIBEResearch — Full Pipeline (Inference + Evaluation)
#
# Presets (edit MODEL_NAME, VLLM_URL, TOOL_SET, API_KEY as needed):
#   - DeepSeek-V3.2 + builtin:  MODEL_NAME=deepseek-v3.2  TOOL_SET=builtin
#   - Kimi-K2.5 + builtin:      MODEL_NAME=kimi-k2.5      TOOL_SET=builtin  VLLM_URL=<kimi>
#   - Doubao + custom:          MODEL_NAME=doubao-seed     TOOL_SET=custom   API_KEY=$ARK_API_KEY
#   - GLM-5.1 + custom:         MODEL_NAME=glm-5.1         TOOL_SET=custom
#
# Usage:
#   MODEL_NAME=kimi-k2.5 VLLM_URL=http://host/v1 bash scripts/run_all.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"

export no_proxy=""
export NO_PROXY="$no_proxy"
export SERPER_API_KEY="${SERPER_API_KEY:-}"

# ---------- Config: set for your run ----------
MODEL_NAME="${MODEL_NAME:-doubao-seed-2-0-pro-260215}"
VLLM_URL="${VLLM_URL:-https://}"
TOOL_SET="${TOOL_SET:-custom}"
API_KEY="${API_KEY:-}"

# Model config profile (overrides MODEL_NAME/VLLM_URL/API_KEY when set)
MODEL_CONFIG="${MODEL_CONFIG:-}"
MODEL_PROFILE="${MODEL_PROFILE:-}"

# API type: openai / deployed / azure / gemini / claude
API_TYPE="${API_TYPE:-openai}"
API_VERSION="${API_VERSION:-}"
THINKING_BUDGET="${THINKING_BUDGET:-0}"

# For tool_set=custom only
SUMMARIZE_URL="${SUMMARIZE_URL:-http://localhost:80/v1}"
SUMMARIZE_MODEL="${SUMMARIZE_MODEL:-qwen3-30b-a3b-instruct}"
CODE_SANDBOX_URL="${CODE_SANDBOX_URL:-http://localhost:8080/run_code}"

DATA_PATH="${DATA_PATH:-$PROJECT_DIR/tasks/0423_debug_10}"
GEMINI_API_KEY="${GEMINI_API_KEY:-your_api_key_here}"
GEMINI_API_URL="${GEMINI_API_URL:-https://example.com/gemini}"

# Log
LOG_DIR="$PROJECT_DIR/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log}"

# Sampling
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-240000}"

# Multi-agent: set MULTI_AGENT=1 to enable
MULTI_AGENT="${MULTI_AGENT:-0}"

# Execution mode: direct / staged / simulated
MODE="${MODE:-simulated}"
MAX_USER_TURNS="${MAX_USER_TURNS:-50}"
USER_MODEL="${USER_MODEL:-}"
USER_MODEL_URL="${USER_MODEL_URL:-}"
USER_MODEL_API_KEY="${USER_MODEL_API_KEY:-}"

# ---------- Build command ----------
EXTRA=()
[[ -n "${MODEL_CONFIG:-}" ]] && EXTRA+=(--model-config "$MODEL_CONFIG")
[[ -n "${MODEL_PROFILE:-}" ]] && EXTRA+=(--model-profile "$MODEL_PROFILE")

# When using model_config profile, skip params managed by the profile;
# otherwise fall back to shell-level defaults.
if [[ -n "${MODEL_CONFIG:-}" && -n "${MODEL_PROFILE:-}" ]]; then
  :
else
  [[ -n "${API_KEY:-}" ]] && EXTRA+=(--api-key "$API_KEY")
  EXTRA+=(--api-type "$API_TYPE")
  [[ -n "${API_VERSION:-}" ]] && EXTRA+=(--api-version "$API_VERSION")
  [[ "$THINKING_BUDGET" != "0" ]] && EXTRA+=(--thinking-budget "$THINKING_BUDGET")
  EXTRA+=(--temperature "$TEMPERATURE" --top-p "$TOP_P" --max-tokens "$MAX_TOKENS")
fi

[[ "$TOOL_SET" == "custom" ]] && EXTRA+=(--summarize-url "$SUMMARIZE_URL" --summarize-model "$SUMMARIZE_MODEL" --sandbox-url "$CODE_SANDBOX_URL")
[[ "$MULTI_AGENT" == "1" ]] && EXTRA+=(--multi-agent)
EXTRA+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
EXTRA+=(--mode "$MODE")
[[ "$MODE" == "simulated" ]] && EXTRA+=(--max-user-turns "$MAX_USER_TURNS")
[[ -n "${USER_MODEL:-}" ]] && EXTRA+=(--user-model "$USER_MODEL")
[[ -n "${USER_MODEL_URL:-}" ]] && EXTRA+=(--user-model-url "$USER_MODEL_URL")
[[ -n "${USER_MODEL_API_KEY:-}" ]] && EXTRA+=(--user-model-api-key "$USER_MODEL_API_KEY")

DATASET_NAME="$(basename "$DATA_PATH")"
if [[ -n "${MODEL_CONFIG:-}" && -n "${MODEL_PROFILE:-}" ]]; then
  echo ">>> VIBEResearch Full Pipeline (dataset=$DATASET_NAME profile=$MODEL_PROFILE tool_set=$TOOL_SET multi_agent=$MULTI_AGENT mode=$MODE)"
  echo ">>> Log: $LOG_FILE"
  $PYTHON run.py \
    --data-path "$DATA_PATH" \
    --num-samples 4 \
    --tool-set "$TOOL_SET" \
    --browser-backend serper \
    --grader-type gemini \
    --grader-api-url "$GEMINI_API_URL" \
    --grader-api-key "$GEMINI_API_KEY" \
    --max-concurrency "${MAX_CONCURRENCY:-20}" \
    --grader-threads "${GRADER_THREADS:-20}" \
    "${EXTRA[@]}" \
    2>&1 | tee "$LOG_FILE"
else
  echo ">>> VIBEResearch Full Pipeline (dataset=$DATASET_NAME model=$MODEL_NAME api_type=$API_TYPE tool_set=$TOOL_SET multi_agent=$MULTI_AGENT mode=$MODE)"
  echo ">>> Log: $LOG_FILE"
  $PYTHON run.py \
    --data-path "$DATA_PATH" \
    --num-samples 4 \
    --vllm-server-url "$VLLM_URL" \
    --model "$MODEL_NAME" \
    --tool-set "$TOOL_SET" \
    --browser-backend serper \
    --grader-type gemini \
    --grader-api-url "$GEMINI_API_URL" \
    --grader-api-key "$GEMINI_API_KEY" \
    --max-concurrency "${MAX_CONCURRENCY:-20}" \
    --grader-threads "${GRADER_THREADS:-20}" \
    "${EXTRA[@]}" \
    2>&1 | tee "$LOG_FILE"
fi
