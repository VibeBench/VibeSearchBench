#!/usr/bin/env python3
"""VIBEResearch benchmark runner.

Pipeline:
  1. Load task data from JSON files
  2. Run agent (single or multi-agent mode) to generate trajectories
  3. Grade responses with LLM judge
  4. Report metrics

Usage:
  # Inference + evaluation
  python run.py --model glm-5.1 --vllm-server-url http://host/v1

  # Inference only
  python run.py --model glm-5.1 --vllm-server-url http://host/v1 --skip-eval

  # Evaluation only (on existing trajectories)
  python run.py --eval-only --trajs-dir results/trajs/glm-5.1_custom_serper \\
      --grader-type gemini --grader-api-url ...
"""

import argparse
import asyncio
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="VIBEResearch benchmark: agent inference + KG evaluation"
    )

    # --- Data ---
    p.add_argument("--data-path", default="./tasks",
                    help="Path to task JSON files (default: ./tasks)")
    p.add_argument("-n", "--limit", type=int, default=None,
                    help="Max number of tasks to process")
    p.add_argument("--num-samples", type=int, default=4,
                    help="Number of independent trajectories per query (default: 4)")

    # --- Agent ---
    p.add_argument("--agent-type", default="general",
                    help="Agent type: general (LLM-based) or openclaw (CLI-based)")
    p.add_argument("--vllm-server-url", default=None,
                   help="Base URL for chat API (e.g. http://host/v1)")
    p.add_argument("--model", default=None,
                    help="Model name for chat API")
    p.add_argument("--api-key", default=None,
                   help="API key for main model")
    p.add_argument("--tool-set", default="custom", choices=["custom", "builtin"],
                    help="Tool set: custom (search/visit/python) or builtin (search/open/find)")
    p.add_argument("--browser-backend", default="serper", choices=["local", "serper"])
    p.add_argument("--search-url", default="http://localhost:8000")
    p.add_argument("--multi-agent", dest="multi_agent", action="store_true",
                    help="Enable create_sub_agents tool for multi-agent mode")
    p.add_argument("--no-system-prompt", dest="no_system_prompt", action="store_true",
                    help="Disable the default system prompt")

    # --- Agent sampling ---
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    p.add_argument("--max-tokens", type=int, default=16384,
                    help="Max output tokens per LLM call")
    p.add_argument("--max-context-tokens", type=int, default=240000,
                    help="Prompt token limit before forcing a final answer")
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--max-concurrency", type=int, default=64)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--max-consecutive-errors", type=int, default=10)
    p.add_argument("--reasoning-effort", default=None,
                    choices=["high", "medium", "low", "xhigh"])
    p.add_argument("--api-type", default="openai",
                    choices=["openai", "deployed", "azure", "gemini", "claude"],
                    help="LLM API type (default: openai)")
    p.add_argument("--api-version", default=None,
                    help="API version for Azure endpoints")
    p.add_argument("--thinking-budget", type=int, default=0,
                    help="Thinking budget tokens for Claude (0 = disabled)")
    p.add_argument("--thinking-level", type=str, default="",
                    help="Thinking level for Gemini 3+ (low/medium/high)")

    # --- Custom tool set options ---
    p.add_argument("--summarize-url", default=None,
                    help="vLLM URL for visit summarization")
    p.add_argument("--summarize-model", default=None,
                    help="Model name for visit summarization")
    p.add_argument("--sandbox-url", default=None,
                    help="HTTP code-sandbox URL for python tool")

    # --- OpenClaw agent options ---
    p.add_argument("--gateway-port", type=int, default=18789,
                    help="OpenClaw gateway port (default: 18789)")
    p.add_argument("--source-dir", default="./openclaw_backup",
                    help="OpenClaw source config directory")
    p.add_argument("--openclaw-results-dir", default=None,
                    help="OpenClaw per-task state directory (default: results/{dataset}/{experiment}/workspaces/)")
    p.add_argument("--idle-threshold", type=int, default=90,
                    help="Seconds to wait before nudging openclaw (default: 90)")
    p.add_argument("--max-nudge", type=int, default=3,
                    help="Max nudge attempts when openclaw is idle (default: 3)")
    p.add_argument("--openclaw-model", default=None,
                    help="Model for openclaw (written to openclaw.json agents.defaults.model.primary)")

    # --- Grader ---
    p.add_argument("--grader-type", default="gemini",
                    choices=["openai", "gemini"])
    p.add_argument("--grader-base-url", default=None,
                    help="OpenAI grader: base URL")
    p.add_argument("--grader-api-url", default=None,
                    help="Gemini grader: full API URL")
    p.add_argument("--grader-api-key", default=None)
    p.add_argument("--grader-model", default="gpt-4.1-2025-04-14")
    p.add_argument("--grader-threads", type=int, default=16)

    # --- Output ---
    p.add_argument("--output-dir", default=None,
                    help="Override output directory (default: auto-generated under results/)")
    p.add_argument("--eval-dir", default=None,
                    help="Override eval output directory (default: derived from trajs-dir or output-dir)")

    # --- Execution mode ---
    p.add_argument("--mode", default="direct",
                    choices=["direct", "staged", "simulated"],
                    help="Execution mode: direct (single query), staged (sequential sub-queries), "
                         "simulated (LLM user simulator)")
    p.add_argument("--max-user-turns", type=int, default=30,
                    help="Max user agent turns in simulated mode (default: 30)")
    p.add_argument("--user-model", default=None,
                    help="Model name for user simulator (default: same as main model)")
    p.add_argument("--user-model-url", default=None,
                    help="Base URL for user simulator model (default: same as main)")
    p.add_argument("--user-model-api-key", default=None,
                    help="API key for user simulator model")

    # --- Model config from YAML ---
    p.add_argument("--model-config", default=None,
                    help="Path to model_config.yaml")
    p.add_argument("--model-profile", default=None,
                    help="Profile name in model_config.yaml (e.g. seed)")

    # --- Mode control ---
    p.add_argument("--skip-eval", action="store_true",
                    help="Skip evaluation, only run inference")
    p.add_argument("--eval-only", action="store_true",
                    help="Skip inference, only run evaluation on existing trajectories")
    p.add_argument("--trajs-dir", default=None,
                    help="Path to existing trajectories directory (for --eval-only)")

    return p.parse_args()


def _load_traces_from_dir(traces_dir: str) -> dict:
    """Load all traces from per-query JSONL files under *traces_dir*.

    Each file is {qid}.jsonl; each line is one trace (one sample).
    Returns dict keyed by ``(qid, sample_idx)``.
    """
    traces = {}
    if not os.path.isdir(traces_dir):
        return traces
    for name in os.listdir(traces_dir):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(traces_dir, name)
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
                    key = (str(t["qid"]), t.get("sample_idx", 0))
                    traces[key] = t
        except (json.JSONDecodeError, KeyError):
            raise ValueError(f"Invalid trace file: {path}")
    return traces


def _populate_data(data, all_traces, num_samples):
    """Populate each data item with lists of per-sample responses."""
    for item in data:
        qid_str = str(item["qid"])
        responses = []
        all_messages = []
        for si in range(num_samples):
            t = all_traces.get((qid_str, si), {})
            msgs = t.get("messages", [])
            resp = t.get("response", "")
            all_messages.append(msgs)
            responses.append(resp)
        item["responses"] = responses
        item["all_messages"] = all_messages
        item["messages"] = all_messages[0] if all_messages else []
        item["response"] = responses[0] if responses else ""


def _merge_ratings_into_traces(all_traces: dict, per_item: list) -> None:
    """Merge grading result keys into each trace in place."""
    for row in per_item:
        key = (str(row.get("qid", "")), row.get("sample_idx", 0))
        if key not in all_traces:
            continue
        for k, v in row.items():
            all_traces[key][k] = v


def _write_traces_to_dir(traces_dir: str, all_traces: dict) -> None:
    """Write traces to traces_dir: one file per qid, one JSON line per sample."""
    by_qid = {}
    for (qid, si), trace in all_traces.items():
        by_qid.setdefault(qid, []).append((si, trace))
    for qid, pairs in by_qid.items():
        path = os.path.join(traces_dir, f"{qid}.jsonl")
        pairs.sort(key=lambda x: x[0])
        with open(path, "w", encoding="utf-8") as f:
            for _, t in pairs:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")


def _build_experiment_name(args) -> str:
    """Build experiment name from model and config.

    Always includes the mode suffix so that different modes produce
    distinct output directories, e.g.:
        glm-5.1_custom_serper_direct
        glm-5.1_custom_serper_multi_agent_direct
        openclaw_simulated
    """
    agent_type = getattr(args, "agent_type", "general")
    if agent_type == "openclaw":
        name = "openclaw"
    else:
        model_slug = (args.model or "unknown").replace("/", "_")
        name = f"{model_slug}_{args.tool_set}_{args.browser_backend}"
        if args.multi_agent:
            name += "_multi_agent"
    name += f"_{args.mode}"
    return name


def _dataset_name(data_path: str) -> str:
    """Extract dataset name from the data path.

    Examples:
        ./tasks           -> tasks
        /abs/path/my_data -> my_data
    """
    return os.path.basename(os.path.normpath(data_path))


_PROFILE_KEY_MAP = {
    "model":           ("model",           None),
    "base_url":        ("vllm_server_url", None),
    "api_key":         ("api_key",         None),
    "temperature":     ("temperature",     1.0),
    "top_p":           ("top_p",           0.95),
    "max_tokens":      ("max_tokens",      16384),
    "api_type":        ("api_type",        "openai"),
    "api_version":     ("api_version",     None),
    "thinking_budget":   ("thinking_budget",   0),
    "thinking_level":    ("thinking_level",    ""),
    "reasoning_effort":  ("reasoning_effort",  None),
}

_DEFAULTS_KEY_MAP = {
    "user_model":         ("user_model",         None),
    "user_model_url":     ("user_model_url",     None),
    "user_model_api_key": ("user_model_api_key", None),
}


def _apply_profile(args):
    """Apply model profile values, overriding CLI args when profile is set."""
    if not args.model_config or not args.model_profile:
        return

    from agent.llm import load_profile, load_defaults
    profile = load_profile(args.model_config, args.model_profile)
    logger.info("Loaded model profile %r from %s", args.model_profile, args.model_config)

    for yaml_key, (arg_name, _cli_default) in _PROFILE_KEY_MAP.items():
        yaml_val = profile.get(yaml_key)
        if yaml_val is None or yaml_val == "":
            continue
        setattr(args, arg_name, yaml_val)

    defaults = load_defaults(args.model_config)
    for yaml_key, (arg_name, _cli_default) in _DEFAULTS_KEY_MAP.items():
        yaml_val = defaults.get(yaml_key)
        if yaml_val is None or yaml_val == "":
            continue
        if getattr(args, arg_name, None) is None:
            setattr(args, arg_name, yaml_val)


def main():
    args = parse_args()
    _apply_profile(args)

    from eval.evaluator import load_data, grade, grade_one
    from eval.grader import create_grader
    from agent.prompts import (
        get_developer_content, get_triple_request_prompt,
        SYSTEM_PROMPT_EN, MULTI_AGENT_PROMPT_EN,
    )

    # Determine output directories
    # Structure: results/{dataset}/{experiment}/trajs/  and  .../eval/
    dataset = _dataset_name(args.data_path)
    experiment_name = _build_experiment_name(args) if not args.output_dir else None

    if args.output_dir:
        trajs_dir = os.path.join(args.output_dir, "trajs")
        eval_dir = os.path.join(args.output_dir, "eval")
    else:
        base = os.path.join("results", dataset, experiment_name)
        trajs_dir = os.path.join(base, "trajs")
        eval_dir = os.path.join(base, "eval")

    if args.trajs_dir:
        trajs_dir = args.trajs_dir
        # Auto-derive eval_dir from trajs_dir when no explicit output-dir
        if not args.output_dir and not args.eval_dir:
            # .../trajs -> .../eval
            trajs_norm = os.path.normpath(trajs_dir)
            if trajs_norm.endswith(os.sep + "trajs") or trajs_norm.endswith("/trajs"):
                eval_dir = trajs_norm[:-5] + "eval"
            else:
                eval_dir = trajs_dir + "_eval"

    if args.eval_dir:
        eval_dir = args.eval_dir

    # Load task data
    data = load_data(args.data_path, limit=args.limit)
    logger.info("Loaded %d tasks from %s", len(data), args.data_path)

    num_samples = args.num_samples
    mode = args.mode

    # ================================================================
    # Phase 1: Agent Inference
    # ================================================================
    if not args.eval_only:
        agent_type = getattr(args, "agent_type", "general")
        if agent_type != "openclaw" and (not args.vllm_server_url or not args.model):
            raise ValueError("--vllm-server-url and --model are required for inference (general agent)")

        from agent import create_agent

        os.makedirs(trajs_dir, exist_ok=True)

        # Resume: load already-completed (qid, sample_idx) pairs
        completed_keys: set = set()
        existing = _load_traces_from_dir(trajs_dir)
        if existing:
            for t in existing.values():
                completed_keys.add(
                    (str(t["qid"]), t.get("sample_idx", 0))
                )
            logger.info("Resuming: %d traces already in %s",
                        len(completed_keys), trajs_dir)

        # Expand items x num_samples, skipping completed ones
        items_for_agent = []
        use_multi = getattr(args, "multi_agent", False)
        for it in data:
            for si in range(num_samples):
                if (str(it["qid"]), si) not in completed_keys:
                    item = {
                        "qid": it["qid"],
                        "question": it["question"],
                        "sample_idx": si,
                        "mode": mode,
                    }
                    # Per-item system prompt based on language and mode
                    lang = it.get("language", "en")
                    if not getattr(args, "no_system_prompt", False):
                        item["system_prompt"] = get_developer_content(
                            lang, multi_agent=use_multi, mode=mode,
                        )
                        if use_multi:
                            item["sub_agent_system_prompt"] = get_developer_content(
                                lang, multi_agent=False, mode=mode,
                            )

                    # Triple request prompt for all modes
                    item["triple_request_prompt"] = get_triple_request_prompt(lang)

                    # Mode-specific fields
                    if mode == "staged":
                        item["initial_query"] = it["initial_query"]
                        item["sub_queries"] = it["sub_queries"]
                    elif mode == "simulated":
                        item["initial_query"] = it["initial_query"]
                        item["user_persona"] = it.get("user_persona", "")
                        item["max_user_turns"] = args.max_user_turns
                        if args.user_model:
                            item["user_model_name"] = args.user_model
                        if args.user_model_url:
                            item["user_model_url"] = args.user_model_url
                        if args.user_model_api_key:
                            item["user_model_api_key"] = args.user_model_api_key

                    items_for_agent.append(item)

        if items_for_agent:
            developer_content = MULTI_AGENT_PROMPT_EN if use_multi else SYSTEM_PROMPT_EN
            if getattr(args, "no_system_prompt", False):
                developer_content = None
            sub_agent_dc = SYSTEM_PROMPT_EN if use_multi else None

            if agent_type == "openclaw":
                openclaw_workdir = args.openclaw_results_dir
                if not openclaw_workdir:
                    openclaw_workdir = os.path.join(
                        os.path.dirname(trajs_dir), "workspaces",
                    )
                agent = create_agent(
                    "openclaw",
                    gateway_port=args.gateway_port,
                    source_dir=args.source_dir,
                    openclaw_results_dir=openclaw_workdir,
                    idle_threshold=args.idle_threshold,
                    max_nudge=args.max_nudge,
                    openclaw_model=args.openclaw_model,
                    base_url=args.vllm_server_url,
                    api_key=args.api_key,
                    model_name=args.model,
                )
            else:
                agent = create_agent(
                    agent_type,
                    base_url=args.vllm_server_url,
                    model_name=args.model,
                    api_key=args.api_key,
                    browser_backend=args.browser_backend,
                    search_url=args.search_url,
                    tool_set=args.tool_set,
                    reasoning_effort=args.reasoning_effort,
                    developer_content=developer_content,
                    sub_agent_developer_content=sub_agent_dc,
                    max_rounds=args.max_rounds,
                    max_retries=args.max_retries,
                    max_context_tokens=args.max_context_tokens,
                    max_consecutive_errors=args.max_consecutive_errors,
                    sandbox_url=args.sandbox_url,
                    summarize_url=args.summarize_url,
                    summarize_model=args.summarize_model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    multi_agent=use_multi,
                    api_type=args.api_type,
                    api_version=args.api_version,
                    thinking_budget=args.thinking_budget,
                    thinking_level=args.thinking_level,
                )
            logger.info(
                "Generating %d runs (%d queries x %d samples): "
                "agent=%s model=%s mode=%s",
                len(items_for_agent), len(data), num_samples,
                agent_type, args.model or "openclaw", mode,
            )
            concurrency = 1 if agent_type == "openclaw" else args.max_concurrency
            asyncio.run(
                agent.run_batch(
                    items_for_agent,
                    max_concurrency=concurrency,
                    traces_dir=trajs_dir,
                )
            )
        else:
            logger.info("All %d runs already completed, skipping generation",
                        len(data) * num_samples)

    # ================================================================
    # Phase 2: Evaluation
    # ================================================================
    if args.skip_eval:
        logger.info("Skipping evaluation (--skip-eval)")
        logger.info("Trajectories saved to %s", trajs_dir)
        return

    # Load all traces and populate data
    all_traces = _load_traces_from_dir(trajs_dir)
    if not all_traces:
        logger.error("No traces found in %s. Run inference first.", trajs_dir)
        return

    _populate_data(data, all_traces, num_samples)

    # Summarise termination reasons
    term_counts: dict = {}
    for t in all_traces.values():
        reason = t.get("termination", "unknown")
        term_counts[reason] = term_counts.get(reason, 0) + 1
    if term_counts:
        logger.info("Termination reasons: %s", term_counts)

    total_responses = sum(
        sum(1 for r in d.get("responses", []) if r)
        for d in data
    )
    total_runs = len(data) * num_samples
    logger.info("Responses: %d/%d non-empty", total_responses, total_runs)

    # --- Grading ---
    api_key = args.grader_api_key
    if api_key is None:
        api_key = (
            os.getenv("GEMINI_API_KEY", "")
            if args.grader_type == "gemini"
            else os.getenv("OPENAI_API_KEY", "EMPTY")
        )

    grader_config = {
        "type": args.grader_type,
        "base_url": args.grader_base_url or (args.vllm_server_url or ""),
        "api_url": args.grader_api_url,
        "api_key": api_key,
        "model": args.grader_model,
        "max_retries": args.max_retries,
        "max_workers": args.grader_threads,
    }
    grader_label = (
        f"Gemini @ {args.grader_api_url}"
        if args.grader_type == "gemini"
        else f"{args.grader_model} @ {grader_config['base_url']}"
    )
    logger.info("Grading with %s (%d threads)", grader_label, args.grader_threads)

    os.makedirs(eval_dir, exist_ok=True)

    # Grade and save per-item results
    grader_client = create_grader(grader_config)
    per_item_results = []
    metrics_keys = ["node_precision", "node_recall", "node_f1",
                    "triplet_precision", "triplet_recall", "triplet_f1"]

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm as tqdm_bar

    # Flatten items x samples
    flat = []
    for item in data:
        responses = item.get("responses", [item.get("response", "")])
        for si, resp in enumerate(responses):
            flat.append({
                "qid": item["qid"],
                "response": resp,
                "answer": item["answer"],
                "sample_idx": si,
            })

    def _grade_and_save(fi):
        result = grade_one(
            grader_client,
            fi["qid"],
            fi.get("sample_idx", 0),
            fi.get("response", ""),
            fi["answer"],
        )
        # Save per-trajectory eval result
        safe_qid = str(fi["qid"]).replace("/", "_").replace("\\", "_")
        eval_file = os.path.join(eval_dir, f"{safe_qid}_sample{fi['sample_idx']}.json")
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        return result

    with ThreadPoolExecutor(max_workers=grader_config.get("max_workers", 4)) as pool:
        per_item_results = list(tqdm_bar(
            pool.map(_grade_and_save, flat),
            total=len(flat), desc="VIBEResearch grading"
        ))

    # Group by qid and compute aggregate metrics
    by_qid = {}
    for r in per_item_results:
        by_qid.setdefault(r["qid"], []).append(r)

    num_samples_actual = max((len(v) for v in by_qid.values()), default=1)

    avg_vals = {k: [] for k in metrics_keys}
    best_vals = {k: [] for k in metrics_keys}
    per_query_scores = []
    for qid, samples in sorted(by_qid.items()):
        q_avg = {}
        for k in metrics_keys:
            sv = [s[k] for s in samples]
            v = sum(sv) / len(sv)
            avg_vals[k].append(v)
            q_avg[k] = v
        best = max(samples, key=lambda s: s["triplet_f1"])
        q_best = {}
        for k in metrics_keys:
            best_vals[k].append(best[k])
            q_best[k] = best[k]
        per_query_scores.append({
            "qid": qid,
            "num_samples": len(samples),
            "avg": q_avg,
            "best": q_best,
        })

    summary = {}
    for k in metrics_keys:
        summary[f"avg@{num_samples_actual}_{k}"] = round(sum(avg_vals[k]) / len(avg_vals[k]), 4) if avg_vals[k] else 0.0
        summary[f"best@{num_samples_actual}_{k}"] = round(sum(best_vals[k]) / len(best_vals[k]), 4) if best_vals[k] else 0.0
    summary["num_queries"] = len(by_qid)
    summary["num_samples"] = num_samples_actual

    # Save summary
    summary_path = os.path.join(eval_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save full per-item ratings
    items_path = os.path.join(eval_dir, "item_ratings.json")
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(per_item_results, f, indent=2, ensure_ascii=False, default=str)

    # Save per-query scores
    per_query_path = os.path.join(eval_dir, "per_query_scores.json")
    with open(per_query_path, "w", encoding="utf-8") as f:
        json.dump(per_query_scores, f, indent=2, ensure_ascii=False)

    # Merge grading results into trace files
    if os.path.isdir(trajs_dir):
        _merge_ratings_into_traces(all_traces, per_item_results)
        _write_traces_to_dir(trajs_dir, all_traces)
        logger.info("Traces updated with ratings in %s", trajs_dir)

    logger.info("Evaluation results saved to %s", eval_dir)
    logger.info("Trajectories in %s", trajs_dir)

    # ---- Pretty-print results ----
    n = num_samples_actual
    _sep = "=" * 100
    print(f"\n{_sep}")
    print("  VIBEResearch Evaluation Results")
    print(_sep)
    print(f"  Tasks: {len(by_qid)}    Samples/task: {n}    "
          f"Trajs: {trajs_dir}")
    print(f"  Eval dir: {eval_dir}")
    print(_sep)

    # Per-query table
    hdr = (f"  {'Task ID':<45} {'Node F1':>8} {'Trip F1':>8}  |  "
           f"{'Node F1':>8} {'Trip F1':>8}")
    print(f"\n  {'':45} {'--- avg@' + str(n) + ' ---':>18}  |  {'--- best@' + str(n) + ' ---':>18}")
    print(hdr)
    print("  " + "-" * 96)
    for qs in per_query_scores:
        qid_short = str(qs["qid"])
        if len(qid_short) > 43:
            qid_short = qid_short[:40] + "..."
        print(f"  {qid_short:<45} {qs['avg']['node_f1']:>8.4f} {qs['avg']['triplet_f1']:>8.4f}  |  "
              f"{qs['best']['node_f1']:>8.4f} {qs['best']['triplet_f1']:>8.4f}")
    print("  " + "-" * 96)

    # Aggregate summary
    def _g(prefix, key):
        return summary.get(f"{prefix}@{n}_{key}", 0.0)

    print(f"\n  {'AGGREGATE':<45} {_g('avg', 'node_f1'):>8.4f} {_g('avg', 'triplet_f1'):>8.4f}  |  "
          f"{_g('best', 'node_f1'):>8.4f} {_g('best', 'triplet_f1'):>8.4f}")
    print()

    # Detailed aggregate table
    print(f"  {'Metric':<25} {'avg@' + str(n):>10} {'best@' + str(n):>10}")
    print("  " + "-" * 47)
    display_names = {
        "node_precision": "Node Precision",
        "node_recall": "Node Recall",
        "node_f1": "Node F1",
        "triplet_precision": "Triplet Precision",
        "triplet_recall": "Triplet Recall",
        "triplet_f1": "Triplet F1",
    }
    for k in metrics_keys:
        print(f"  {display_names[k]:<25} {_g('avg', k):>10.4f} {_g('best', k):>10.4f}")
    print()
    print(f"  Summary saved to: {summary_path}")
    print(f"  Per-query scores: {per_query_path}")
    print(f"  Item ratings:     {items_path}")
    print(_sep)


if __name__ == "__main__":
    main()
