<p align="center">
  <a href="https://vibebench.github.io/VibeSearchBench.github.io/">
    <img src="assets/img/logo.png" alt="VibeSearchBench" width="220" />
  </a>
</p>

<h1 align="center">VibeSearchBench</h1>

<p align="center"><em>Proactive Search · Evolving Intent · Structured Knowledge</em></p>

<p align="center">
  <a href="https://vibebench.github.io/VibeSearchBench.github.io/"><img src="https://img.shields.io/badge/🌐-Project_Page-2563eb?style=for-the-badge" alt="Project Page" /></a>
  <a href="https://vibebench.github.io/VibeSearchBench.github.io/leaderboard.html"><img src="https://img.shields.io/badge/🏆-Leaderboard-7c3aed?style=for-the-badge" alt="Leaderboard" /></a>
  <a href="https://vibebench.github.io/VibeSearchBench.github.io/assets/paper.pdf"><img src="https://img.shields.io/badge/📄-Paper-18181b?style=for-the-badge" alt="Paper" /></a>
  <a href="https://huggingface.co/datasets/VibeSearchBench/VibeSearchBench"><img src="https://img.shields.io/badge/🤗-Dataset-ffd21e?style=for-the-badge" alt="Dataset" /></a>
</p>

<p align="center" style="margin-top:1.1em;margin-bottom:0.35em">
  <strong>
    By far the
    <span style="color:#dc2626;background:rgba(220,38,38,0.12);padding:0.15em 0.45em;border-radius:5px;font-weight:800;border-bottom:2px solid rgba(220,38,38,0.45)">hardest</span>
    <span style="color:#15803d;background:rgba(22,163,74,0.12);padding:0.15em 0.45em;border-radius:5px;font-weight:800;border-bottom:2px solid rgba(22,163,74,0.45)">verifiable</span>
    <span style="color:#7c3aed;background:rgba(124,58,237,0.12);padding:0.15em 0.45em;border-radius:5px;font-weight:800;border-bottom:2px solid rgba(124,58,237,0.45)">long-horizon</span>
    search benchmark
  </strong>
</p>

<p align="center" style="color:#71717a;font-size:0.92em;line-height:1.55;margin:0.2em 0 0.65em">
  200 bilingual tasks · proactive search in the wild · persona-driven progressive disclosure · schema-free knowledge graph evaluation
</p>

<p align="center" style="margin-bottom:1.75em">
  <img src="https://img.shields.io/badge/Tasks-200-2563eb?style=flat-square" alt="200 Tasks" />
  <img src="https://img.shields.io/badge/Domains-20-0891b2?style=flat-square" alt="20 Domains" />
  <img src="https://img.shields.io/badge/Models-7-7c3aed?style=flat-square" alt="7 Models evaluated" />
  <img src="https://img.shields.io/badge/Best_Triplet_F1-30.3-16a34a?style=flat-square" alt="Best triplet F1 30.3" />
</p>

Official code for **[VibeSearchBench](https://vibebench.github.io/VibeSearchBench.github.io/)** — benchmarking long-horizon proactive search with persona-driven multi-turn interaction and schema-free knowledge graph evaluation.

Real users rarely specify full intent upfront. **VibeSearch** captures bidirectional convergence: agents interleave partial results with follow-up questions while users progressively disclose needs. This repo provides agents, evaluation, and scripts to run the benchmark locally.

| Subset | Description |
|--------|-------------|
| **VibeSearch-Pro** | 100 professional research scenarios — literature reviews, market analysis, technical due diligence |
| **VibeSearch-Daily** | 100 daily-life search tasks — shopping, travel, lifestyle with vague initial queries |

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

<p align="center">
  VibeSearchBench · Rednote-Hilab &amp; Unipat AI
</p>
