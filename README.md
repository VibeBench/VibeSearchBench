<div align="center">

<img src="assets/img/logo.png" width="160" alt="VibeSearchBench Logo">

# VibeSearchBench

[![Tasks](https://img.shields.io/badge/tasks-200-blue)](#tasks)
[![Domains](https://img.shields.io/badge/domains-20-blue)](#tasks)
[![Models](https://img.shields.io/badge/models-7-green)](#leaderboard)
[![Best F1](https://img.shields.io/badge/best_triplet_F1-30.3-green)](#leaderboard)
[![Paper](https://img.shields.io/badge/paper-PDF-red)](https://vibebench.github.io/VibeSearchBench.github.io/assets/paper.pdf)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-purple)](https://vibebench.github.io/VibeSearchBench.github.io/leaderboard.html)
[![Project Page](https://img.shields.io/badge/project_page-live-2563eb)](https://vibebench.github.io/VibeSearchBench.github.io/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/VibeSearchBench/VibeSearchBench)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

> By far the
> <span style="color:#dc2626;background:rgba(220,38,38,0.12);padding:0.1em 0.35em;border-radius:4px;font-weight:800">hardest</span>
> <span style="color:#15803d;background:rgba(22,163,74,0.12);padding:0.1em 0.35em;border-radius:4px;font-weight:800">verifiable</span>
> <span style="color:#7c3aed;background:rgba(124,58,237,0.12);padding:0.1em 0.35em;border-radius:4px;font-weight:800">long-horizon</span>
> search benchmark. <br>
> 200 bilingual tasks | 20 domains | persona-driven progressive disclosure | schema-free knowledge graph evaluation.

</div>

---

## Leaderboard

Browse the full leaderboard and multi-turn task trajectories at **[vibebench.github.io/VibeSearchBench.github.io](https://vibebench.github.io/VibeSearchBench.github.io/)**.

**Evaluation:**

* **Primary metric: Triplet F1.** Predicted knowledge graphs are matched against ground truth via LLM-as-judge node alignment and triplet semantic equivalence.
* **Multi-turn interaction.** Each task uses a persona-driven user simulator with progressive disclosure; agents may search, visit pages, and run code across many turns.
* **Best reported score:** **30.3** triplet F1 (Claude Opus 4.6, OpenClaw).

## Tasks

200 tasks across 2 subsets and 20 domains. Each task pairs a vague initial query with a ground-truth knowledge graph.

| Split | Count | Description |
|-------|-------|-------------|
| `pro` | 100 | Professional research — literature reviews, market analysis, technical due diligence |
| `daily` | 100 | Daily-life search — shopping, travel, lifestyle with evolving preferences |

Real users rarely specify full intent upfront. **VibeSearch** captures bidirectional convergence: agents interleave partial results with follow-up questions while users progressively disclose needs.

### Dataset

Available on Hugging Face: [VibeSearchBench/VibeSearchBench](https://huggingface.co/datasets/VibeSearchBench/VibeSearchBench)

| Field | Description |
|-------|-------------|
| `qid` | Unique task identifier |
| `question` | Full research query with constraints |
| `user_persona` | Persona for the progressive-disclosure simulator |
| `nodes` / `triples` | Ground-truth knowledge graph |

---

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

---

## Project Structure

```
VibeSearchBench/
├── agent/                          # Agent implementations
│   ├── general_agent.py            # GeneralAgent (OpenAI-compatible, single/multi-agent)
│   ├── openclaw_agent.py           # OpenClaw agent wrapper
│   ├── llm.py                      # LLM client utilities
│   ├── prompts.py                  # Prompt templates
│   └── toolkit.py                  # ToolKit (search / visit / python via Serper)
├── eval/                           # Evaluation module
│   ├── grader.py                   # GraderClient (OpenAI / Gemini backends)
│   └── evaluator.py                # KG evaluation: node F1, triplet F1
├── scripts/                        # Bash/Python scripts
│   ├── run_all.sh                  # Full pipeline (inference + evaluation)
│   ├── run_inference.sh            # Agent inference only
│   ├── run_eval.sh                 # Evaluation only
│   ├── run_openclaw.sh             # OpenClaw evaluation
│   └── build_website_data.py       # Export data for the project page
├── viberesearch_query_synthesis/   # Query synthesis module
├── website/                        # Static site template (deployed via github.io repo)
├── tasks/                          # Task JSON files (benchmark data)
├── results/                        # Output (auto-created)
├── model_config.yaml               # LLM model profiles
└── run.py                          # Main entry point
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

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">VibeSearchBench · Rednote-Hilab &amp; Unipat AI</p>
