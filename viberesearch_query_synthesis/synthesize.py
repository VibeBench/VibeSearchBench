"""
Query synthesis script: generates final_query, initial_query, and user_persona
for each task JSON using an OpenAI-compatible LLM API.

Usage:
    python synthesize.py \
        --task-dir ../tasks/sample10 \
        --output-dir ./output \
        --base-url http://your-vllm-endpoint/v1 \
        --model your-model-name
"""

import json
import glob
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

import sys

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.llm import get_sync_client
from prompts import get_final_query_prompt, get_initial_query_prompt, get_user_persona_prompt


def build_node_map(nodes: List[Dict]) -> Dict[str, str]:
    return {n["node_id"]: n["node_name"].strip() for n in nodes}


def graph_to_text(nodes: List[Dict], triples: List[Dict], language: str = "zh") -> str:
    node_map = build_node_map(nodes)
    lines = []
    for t in triples:
        head = node_map.get(t["head_id"], t["head_id"])
        tail = node_map.get(t["tail_id"], t["tail_id"])
        rel = t["relation"]
        lines.append(f"- {head} —{rel}→ {tail}")
    if not lines:
        return "(No knowledge graph data)" if language == "en" else "（无知识图谱数据）"
    return "\n".join(lines)


def format_user_queries(user_queries: List[str], language: str = "zh") -> str:
    parts = []
    for i, q in enumerate(user_queries, 1):
        if language == "en":
            parts.append(f"Query {i}: {q}")
        else:
            parts.append(f"第{i}轮查询：{q}")
    return "\n".join(parts)


def call_llm(client: OpenAI, model: str, prompt: str,
             temperature: float = 0.7, max_tokens: int = 4096) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def synthesize_task(client: OpenAI, model: str, data: Dict[str, Any],
                    temperature: float = 0.7, max_tokens: int = 4096,
                    persona_only: bool = False) -> Dict[str, Any]:
    user_queries = data.get("user_queries", [])
    nodes = data.get("nodes", [])
    triples = data.get("triples", [])
    language = data.get("language", "zh")

    user_queries_text = format_user_queries(user_queries, language)
    graph_text = graph_to_text(nodes, triples, language)

    result = dict(data)

    if persona_only:
        initial_query = data.get("initial_query", "")
        final_query = data.get("final_query", "")
    else:
        prompt_final = get_final_query_prompt(language).format(
            user_queries_text=user_queries_text,
            graph_text=graph_text,
        )
        final_query = call_llm(client, model, prompt_final, temperature, max_tokens)

        prompt_initial = get_initial_query_prompt(language).format(
            final_query=final_query,
            user_queries_text=user_queries_text,
        )
        initial_query = call_llm(client, model, prompt_initial, temperature, max_tokens)

        result["final_query"] = final_query
        result["initial_query"] = initial_query

    prompt_persona = get_user_persona_prompt(language).format(
        final_query=final_query,
        initial_query=initial_query,
        user_queries_text=user_queries_text,
        graph_text=graph_text,
    )
    user_persona = call_llm(client, model, prompt_persona, temperature, max_tokens)

    result["user_persona"] = user_persona
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize final_query, initial_query, and user_persona for task JSONs. "
                    "Use --persona-only to only synthesize user_persona."
    )
    parser.add_argument("--task-dir", required=True,
                        help="Directory containing task JSON files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <script_dir>/output)")
    parser.add_argument("--base-url", default="",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default="",
                        help="Model name")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key (default: EMPTY)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-concurrency", type=int, default=5,
                        help="Max number of tasks to process in parallel")
    parser.add_argument("--persona-only", action="store_true",
                        help="Only synthesize user_persona; keep existing initial_query and final_query")
    args = parser.parse_args()

    if not args.base_url or not args.model:
        print("Error: --base-url and --model are required.")
        print("Example: python synthesize.py --task-dir ../tasks/sample10 "
              "--base-url http://localhost:8000/v1 --model qwen2.5-72b")
        return

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    client = get_sync_client(args.base_url, args.api_key)

    json_files = sorted(glob.glob(os.path.join(args.task_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {args.task_dir}")
        return

    print(f"Found {len(json_files)} task files. Output → {output_dir}")
    print(f"Model: {args.model} @ {args.base_url}")
    print(f"Mode: {'persona-only' if args.persona_only else 'full (initial_query + final_query + user_persona)'}")
    print(f"Concurrency: {args.max_concurrency}\n")

    def process_one(fpath: str) -> str:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data.get("user_queries"):
                return f"[skip] {fname}: no user_queries"

            result = synthesize_task(
                client, args.model, data,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                persona_only=args.persona_only,
            )

            out_path = os.path.join(output_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return (f"[done] {fname}\n"
                    f"       final_query: {len(result['final_query'])} chars | "
                    f"initial_query: {len(result['initial_query'])} chars | "
                    f"user_persona: {len(result['user_persona'])} chars")
        except Exception as e:
            return f"[error] {fname}: {e}"

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futures = {pool.submit(process_one, fp): fp for fp in json_files}
        for future in as_completed(futures):
            print(future.result())

    print("\nAll tasks finished.")


if __name__ == "__main__":
    main()
