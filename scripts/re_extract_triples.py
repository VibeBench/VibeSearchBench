#!/usr/bin/env python3
"""Re-extract triples from existing trajectories using the current prompt.

Usage:
    python scripts/re_extract_triples.py \
        --trajs-dir results/vrb_final/claude-opus-4.6_custom_serper_simulated/trajs \
        --output-dir /tmp/re_extracted_triples \
        --model-config model_config.yaml \
        --model-profile claude_opus_4_6 \
        --max-concurrency 10
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
import glob
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.llm import async_chat_completion, load_profile
from agent.prompts import get_triple_request_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRIPLE_MARKERS = ["知识图谱", "knowledge graph", "structured knowledge"]

EVAL_KEYS = {
    "node_precision", "node_recall", "node_f1", "node_details",
    "triplet_precision", "triplet_recall", "triplet_f1",
    "coverage_details", "validity_details", "num_pred_triples",
}


def find_triple_idx(messages: list) -> int:
    """Find the index of the last triple-request user message."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").lower()
        if any(marker in content for marker in TRIPLE_MARKERS):
            return i
    return -1


def detect_language(messages: list) -> str:
    """Detect language from the system prompt."""
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content") or ""
            if "角色" in content or "你是" in content:
                return "zh"
            return "en"
    return "en"


def trace_to_api_msgs(trace: list, question: str) -> list:
    """Convert a trace (with possible compact markers) into api_msgs format.

    Mirrors the rebuild logic in GeneralAgent._compact_context:
    after a [Context compacted] marker, api_msgs becomes
    [system, user(question), assistant(summary), user(continue), ...post-compact msgs].
    """
    last_compact_idx = -1
    for i in range(len(trace) - 1, -1, -1):
        if trace[i].get("role") == "system" and "[Context compacted]" in (trace[i].get("content") or ""):
            last_compact_idx = i
            break

    if last_compact_idx < 0:
        return copy.deepcopy(trace)

    system_msg = None
    if trace[0].get("role") == "system" and "[Context compacted]" not in (trace[0].get("content") or ""):
        system_msg = copy.deepcopy(trace[0])

    summary = (trace[last_compact_idx].get("content") or "").replace("[Context compacted] ", "", 1)

    api_msgs = []
    if system_msg:
        api_msgs.append(system_msg)
    api_msgs.append({"role": "user", "content": question})
    api_msgs.append({"role": "assistant", "content": f"Here is a summary of my previous research:\n\n{summary}"})
    api_msgs.append({"role": "user", "content": "Please continue your research based on the summary above."})

    for m in trace[last_compact_idx + 1:]:
        api_msgs.append(copy.deepcopy(m))

    return api_msgs


async def re_extract_one_sample(
    data: dict,
    cfg: dict,
    sem: asyncio.Semaphore,
    max_tokens: int,
    extra_body: dict | None = None,
) -> dict | None:
    """Re-extract triples for a single sample (one line in the jsonl)."""
    qid = data.get("qid", "?")
    sample_idx = data.get("sample_idx", 0)
    label = f"{qid}_s{sample_idx}"

    msgs = data["messages"]
    question = data.get("question", "")
    triple_idx = find_triple_idx(msgs)

    if triple_idx < 0:
        logger.warning("%s: no triple request found, skipping", label)
        return None

    lang = detect_language(msgs)
    new_prompt = get_triple_request_prompt(lang)

    compact_count = data.get("token_usage", {}).get("compact_count", 0)
    if compact_count > 0:
        context_msgs = trace_to_api_msgs(msgs[:triple_idx], question)
    else:
        context_msgs = copy.deepcopy(msgs[:triple_idx])
    context_msgs.append({"role": "user", "content": new_prompt})

    max_retries = 8
    async with sem:
        t0 = time.time()
        resp = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = await async_chat_completion(
                    cfg, context_msgs,
                    max_tokens=max_tokens,
                    tools=None,
                    tool_choice=None,
                    extra_body=extra_body,
                    n=1,
                    stream=False,
                )
                if not resp.choices:
                    raise RuntimeError("empty response (no choices)")
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = min(2 ** attempt, 60)
                    logger.warning("%s: attempt %d/%d failed (%s), retrying in %ds",
                                   label, attempt, max_retries, e, wait)
                    await asyncio.sleep(wait)
                else:
                    logger.error("%s: all %d attempts failed, last error: %s",
                                 label, max_retries, e)
                    return None
        msg = resp.choices[0].message
        new_response = (msg.content or "").strip()
        usage = resp.usage

    elapsed = round(time.time() - t0, 1)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    logger.info(
        "%s: done in %.1fs, prompt=%d completion=%d response=%d chars",
        label, elapsed, prompt_tokens, completion_tokens, len(new_response),
    )

    out_data = copy.deepcopy(data)
    for k in EVAL_KEYS:
        out_data.pop(k, None)
    out_data["response"] = new_response
    out_msgs = copy.deepcopy(msgs[:triple_idx])
    out_msgs.append({"role": "user", "content": new_prompt})
    out_msgs.append({"role": "assistant", "content": new_response})
    out_data["messages"] = out_msgs
    return out_data


async def re_extract_file(
    traj_path: str,
    output_dir: str,
    cfg: dict,
    sem: asyncio.Semaphore,
    max_tokens: int,
    extra_body: dict | None = None,
):
    """Re-extract triples for all samples in a single jsonl file."""
    fname = os.path.basename(traj_path)
    with open(traj_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    tasks = [
        re_extract_one_sample(s, cfg, sem, max_tokens, extra_body)
        for s in samples
    ]
    results = await asyncio.gather(*tasks)

    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as f:
        for orig, result in zip(samples, results):
            if result is not None:
                out = result
            else:
                out = copy.deepcopy(orig)
                for k in EVAL_KEYS:
                    out.pop(k, None)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")



async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajs-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--model-profile", required=True)
    p.add_argument("--max-concurrency", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=32768)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    profile = load_profile(args.model_config, args.model_profile)
    cfg = {
        "api_type": profile.get("api_type", "openai"),
        "base_url": profile["base_url"],
        "api_key": profile.get("api_key", "EMPTY"),
        "model": profile["model"],
    }
    if profile.get("api_version"):
        cfg["api_version"] = profile["api_version"]
    if profile.get("thinking_budget"):
        cfg["thinking_budget"] = profile["thinking_budget"]
    if profile.get("thinking_level"):
        cfg["thinking_level"] = profile["thinking_level"]

    extra_body = None
    model_lower = profile["model"].lower()
    if "deepseek" in model_lower or "kimi" in model_lower:
        extra_body = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}}

    files = sorted(glob.glob(os.path.join(args.trajs_dir, "*.jsonl")))
    existing = set(os.listdir(args.output_dir))
    files = [f for f in files if os.path.basename(f) not in existing]

    total_samples = 0
    for f in files:
        with open(f) as fh:
            total_samples += sum(1 for line in fh if line.strip())

    logger.info(
        "Re-extracting %d files (%d samples) (profile=%s, concurrency=%d, output=%s)",
        len(files), total_samples, args.model_profile, args.max_concurrency, args.output_dir,
    )

    sem = asyncio.Semaphore(args.max_concurrency)
    tasks = [
        re_extract_file(f, args.output_dir, cfg, sem, args.max_tokens, extra_body)
        for f in files
    ]
    await asyncio.gather(*tasks)

    out_count = len(os.listdir(args.output_dir))
    logger.info("Done. %d files written to %s", out_count, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
