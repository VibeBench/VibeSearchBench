# VIBEResearch Benchmark

Knowledge graph construction benchmark via multi-step web research queries.

## Project Structure

```
viberesearch/
├── agent/                          # Agent implementations
│   ├── __init__.py                 # BaseAgent, registry, create_agent()
│   ├── general_agent.py            # GeneralAgent (OpenAI-compatible, single/multi-agent)
│   ├── llm.py                      # LLM client utilities (async_chat_completion, load_profile)
│   ├── openclaw_agent.py           # OpenClaw agent wrapper
│   ├── prompts.py                  # Prompt templates (system, triple extraction, etc.)
│   └── toolkit.py                  # ToolKit (search/visit/python via Serper API)
├── eval/                           # Evaluation module
│   ├── __init__.py
│   ├── grader.py                   # GraderClient (OpenAI / Gemini backends)
│   └── evaluator.py                # KG evaluation: node F1, triplet F1
├── openclaw/                       # OpenClaw integration
├── prompts/                        # Prompt design docs
│   └── query_synthesis.md          # Query synthesis prompt spec
├── scripts/                        # Bash/Python scripts
│   ├── run_all.sh                  # Full pipeline (inference + evaluation)
│   ├── run_inference.sh            # Agent inference only
│   ├── run_eval.sh                 # Evaluation only (on existing trajectories)
│   ├── run_openclaw.sh             # OpenClaw evaluation
│   ├── run_query_synthesis.sh      # Query synthesis pipeline
│   ├── run_persona_synthesis.sh    # Persona-only synthesis
│   ├── generate_meta_eval_csv.py   # Generate meta-evaluation CSV
│   └── re_extract_triples.py       # Re-extract triples from existing trajectories
├── test/                           # Tests
│   ├── test_llm.py
│   └── test_vllm.py
├── viberesearch_query_synthesis/   # Query synthesis module
│   ├── prompts.py                  # Synthesis prompt templates
│   └── synthesize.py               # Synthesis entry point
├── tasks/                          # Task JSON files (benchmark data)
├── results/                        # Output (auto-created)
│   ├── trajs/                      # Trajectory JSONL files per experiment
│   └── eval/                       # Evaluation results per experiment
├── model_config.yaml               # LLM model profiles (URLs, keys, sampling params)
├── run.py                          # Main entry point
└── README.md
```

## Quick Start

### GeneralAgent (LLM-based)

Uses an OpenAI-compatible LLM to drive multi-step web research.

```bash
# Full pipeline (inference + evaluation)
MODEL_NAME=glm-5.1 VLLM_URL=http://host/v1 bash scripts/run_all.sh

# Inference only
MODEL_NAME=kimi-k2.5 VLLM_URL=http://host/v1 bash scripts/run_inference.sh

# With model config profile
MODEL_CONFIG=model_config.yaml MODEL_PROFILE=seed2_0_pro bash scripts/run_all.sh
```

### OpenClaw Agent (CLI-based)

Wraps the OpenClaw CLI into the benchmark. Requires a running OpenClaw gateway.

```bash
# Default (simulated mode)
bash scripts/run_openclaw.sh

# Direct mode (no user simulation)
MODE=direct bash scripts/run_openclaw.sh

# Custom data and model
DATA_PATH=tasks/my_tasks MODE=simulated OPENCLAW_MODEL=my-model bash scripts/run_openclaw.sh
```

Key OpenClaw env vars: `GATEWAY_PORT` (default 18789), `SOURCE_DIR`, `IDLE_THRESHOLD`, `MAX_NUDGE`, `OPENCLAW_MODEL`.

### Evaluation Only

```bash
TRAJS_DIR=results/trajs/glm-5.1_custom_serper bash scripts/run_eval.sh
```

### Direct Python Usage

```bash
# GeneralAgent: full pipeline
python run.py \
  --agent-type general \
  --model glm-5.1 \
  --vllm-server-url http://host/v1 \
  --tool-set custom \
  --num-samples 4 \
  --grader-type gemini \
  --grader-api-url https://... \
  --grader-api-key YOUR_KEY

# GeneralAgent: inference only
python run.py \
  --agent-type general \
  --model glm-5.1 \
  --vllm-server-url http://host/v1 \
  --skip-eval

# OpenClaw agent
python run.py \
  --agent-type openclaw \
  --gateway-port 18789 \
  --mode simulated \
  --user-model doubao-seed-2-0-pro \
  --user-model-url http://host/v1 \
  --user-model-api-key YOUR_KEY \
  --num-samples 4

# Eval only
python run.py \
  --eval-only \
  --trajs-dir results/trajs/glm-5.1_custom_serper \
  --grader-type gemini \
  --grader-api-url https://...
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MODEL_NAME` | Model name for chat API | `glm-5.1` |
| `VLLM_URL` | Base URL for chat API | (none) |
| `TOOL_SET` | `custom` or `builtin` | `custom` |
| `API_KEY` | API key for main model | (empty) |
| `MULTI_AGENT` | Set to `1` for multi-agent mode | `0` |
| `SERPER_API_KEY` | Serper API key for web search | (preset) |
| `SUMMARIZE_URL` | vLLM URL for page summarization | (preset) |
| `SUMMARIZE_MODEL` | Model for summarization | `qwen3-30b-a3b-instruct` |
| `CODE_SANDBOX_URL` | HTTP sandbox for Python tool | (preset) |
| `GEMINI_API_KEY` | API key for Gemini grader | (preset) |
| `GEMINI_API_URL` | API URL for Gemini grader | (preset) |

### Tool Sets

- **custom** (default): search (Serper) + visit (Serper scrape + LLM summarize) + python (HTTP sandbox)
- **builtin**: search + open + find (requires `gpt_oss` package)

### Agent Modes

- **Single-agent**: One agent handles the entire query
- **Multi-agent** (`MULTI_AGENT=1`): Main agent can spawn sub-agents for parallel research

## Output Format

### Trajectories (`results/trajs/{experiment}/`)

One JSONL file per task (`{task_id}.jsonl`), each line is one sample:

```json
{"qid": "task_042_...", "sample_idx": 0, "question": "...", "messages": [...], "response": "...", "termination": "answer", ...}
```

### Evaluation (`results/eval/{experiment}/`)

- `{task_id}_sample{N}.json` — Per-trajectory evaluation with node/triplet metrics
- `item_ratings.json` — All per-item results
- `summary.json` — Aggregated metrics (avg@N, best@N)

## Dependencies

```
openai aiohttp httpx tqdm transformers json_repair
```

## Evaluation Metrics

Two-phase LLM-as-judge evaluation:

1. **Node matching**: LLM matches predicted entities to ground-truth entities (alias/translation-aware)
2. **Triplet matching**: For matched entity pairs, LLM judges relation semantic equivalence

Metrics: Precision, Recall, F1 at both node and triplet levels, with avg@N and best@N aggregation across samples.
