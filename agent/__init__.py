"""Agent registry, base class, and shared utilities.

Agents handle the iterative research loop: generating text, calling tools,
and accumulating context until a final answer is produced.

Architecture
~~~~~~~~~~~~
* **BaseAgent** owns the concurrent :meth:`run_batch` template — subclasses
  only implement the model-specific :meth:`run_one` and :meth:`extract_response`.
* A lightweight **registry** lets ``run.py`` instantiate agents by name
  without hard-coding imports.

Usage::

    from agent import create_agent

    agent = create_agent("general", base_url=..., model_name=..., ...)
    results = asyncio.run(agent.run_batch(items, max_concurrency=8))
"""

import asyncio
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============================================================================
# Shared constants
# ============================================================================


def summarize_time_stats(raw: dict) -> dict:
    """Collapse per-call timing lists into ``{key: {count, total_time}, ...}``."""
    summary: Dict[str, Any] = {}
    total = 0.0
    for key, times in raw.items():
        if not times:
            continue
        s = sum(times)
        summary[key] = {"count": len(times), "total_time": round(s, 3)}
        total += s
    summary["total_time"] = round(total, 3)
    return summary


# ============================================================================
# Base agent
# ============================================================================


class BaseAgent(ABC):
    """Abstract base for research agents.

    Subclasses implement:
      * :meth:`setup`            – acquire resources (tokenizer, client, ...)
      * :meth:`run_one`          – the model-specific agent loop
      * :meth:`extract_response` – pull final text from a message trace
      * :meth:`teardown`         – release resources

    The shared :meth:`run_batch` provides concurrent execution, incremental
    trace writing, and progress reporting.
    """

    name: str = "Agent"

    # ---- abstract interface ----

    @abstractmethod
    async def setup(self) -> None:
        """Acquire heavy resources (tokenizer, encoding, ...)."""

    @abstractmethod
    async def run_one(self, question: str, qid: Any, **kwargs) -> dict:
        """Execute the agent loop for a single question (direct mode).

        Must return::

            {
                "messages":      list[dict],
                "termination":   "answer"|"max_rounds"|"context_length"|"error_loop",
                "time_stats":    dict,
                "turns":         int,
                "prompt_tokens": int,
            }
        """

    async def run_one_staged(
        self, initial_query: str, sub_queries: List[str], qid: Any, **kwargs,
    ) -> dict:
        """Execute staged mode (sequential sub-queries). Override in subclass."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support staged mode")

    async def run_one_simulated(
        self, initial_query: str, user_persona: str, qid: Any, **kwargs,
    ) -> dict:
        """Execute simulated mode (LLM user simulator). Override in subclass."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support simulated mode")

    @abstractmethod
    def extract_response(self, messages: List[dict]) -> str:
        """Extract the evaluable answer text from a message trace."""

    async def teardown(self) -> None:
        """Release resources.  Override if the agent holds connections."""

    # ---- shared batch execution ----

    async def run_batch(
        self,
        items: List[dict],
        *,
        max_concurrency: int = 8,
        traces_dir: Optional[str] = None,
    ) -> List[dict]:
        """Run the agent on *items* concurrently.

        Each item must contain ``qid`` and ``question``.
        If *traces_dir* is set, each result is appended to that query's JSONL
        file: traces_dir/{qid}.jsonl (one line per sample; resume-safe).
        """
        await self.setup()

        sem = asyncio.Semaphore(max_concurrency)
        _write_lock = asyncio.Lock() if traces_dir else None
        results: List[Optional[dict]] = [None] * len(items)

        def _trace_path(qid) -> str:
            safe_qid = str(qid).replace("/", "_").replace("\\", "_")
            return os.path.join(traces_dir, f"{safe_qid}.jsonl")

        async def _process(idx: int, item: dict):
            async with sem:
                qid = item.get("qid", idx)
                sample_idx = item.get("sample_idx", 0)
                question = item.get("question", "")
                mode = item.get("mode", "direct")
                t0 = time.time()
                try:
                    run_kwargs = {"sample_idx": sample_idx}
                    if item.get("system_prompt"):
                        run_kwargs["system_prompt_override"] = item["system_prompt"]
                    if item.get("sub_agent_system_prompt"):
                        run_kwargs["sub_agent_system_prompt_override"] = item["sub_agent_system_prompt"]

                    if mode == "staged":
                        if item.get("triple_request_prompt"):
                            run_kwargs["triple_request_prompt"] = item["triple_request_prompt"]
                        ar = await self.run_one_staged(
                            item["initial_query"],
                            item.get("sub_queries", []),
                            qid,
                            **run_kwargs,
                        )
                    elif mode == "simulated":
                        if item.get("triple_request_prompt"):
                            run_kwargs["triple_request_prompt"] = item["triple_request_prompt"]
                        for k in ("max_user_turns", "user_model_name",
                                  "user_model_url", "user_model_api_key"):
                            if item.get(k) is not None:
                                run_kwargs[k] = item[k]
                        ar = await self.run_one_simulated(
                            item["initial_query"],
                            item.get("user_persona", ""),
                            qid,
                            **run_kwargs,
                        )
                    else:
                        if item.get("triple_request_prompt"):
                            run_kwargs["triple_request_prompt"] = item["triple_request_prompt"]
                        ar = await self.run_one(question, qid, **run_kwargs)

                    response = self.extract_response(ar["messages"])
                    results[idx] = {
                        "qid": qid,
                        "sample_idx": sample_idx,
                        "question": question,
                        "messages": ar["messages"],
                        "response": response,
                        "termination": ar["termination"],
                        "time_stats": ar["time_stats"],
                        "turns": ar["turns"],
                        "prompt_tokens": ar["prompt_tokens"],
                        "token_usage": ar.get("token_usage"),
                        "latency_s": round(time.time() - t0, 2),
                        "error": None,
                    }
                    logger.info(
                        "qid=%s[%d] done (%d chars, %.1fs, %s, mode=%s)",
                        qid, sample_idx, len(response), time.time() - t0,
                        ar["termination"], mode,
                    )
                except Exception as e:
                    logger.error("qid=%s[%d] failed: %s", qid, sample_idx, e)
                    results[idx] = {
                        "qid": qid,
                        "sample_idx": sample_idx,
                        "question": question,
                        "messages": [],
                        "response": "",
                        "termination": "error",
                        "time_stats": {},
                        "turns": 0,
                        "prompt_tokens": 0,
                        "latency_s": round(time.time() - t0, 2),
                        "error": traceback.format_exc(),
                    }
                if traces_dir and _write_lock and results[idx]:
                    async with _write_lock:
                        path = _trace_path(results[idx]["qid"])
                        with open(path, "a") as f:
                            f.write(
                                json.dumps(results[idx], ensure_ascii=False) + "\n"
                            )

        tasks = [
            asyncio.create_task(_process(i, it)) for i, it in enumerate(items)
        ]
        for fut in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc=self.name
        ):
            await fut

        await self.teardown()
        return results


# ============================================================================
# Registry
# ============================================================================

_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Decorator: register an agent class under *name*."""
    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return _wrap


def create_agent(name: str, **kwargs) -> BaseAgent:
    """Instantiate a registered agent by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown agent '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_agents() -> List[str]:
    return sorted(_REGISTRY.keys())


# Import submodules to trigger @register decorators
from . import general_agent as _general_agent  # noqa: F401,E402
from . import openclaw_agent as _openclaw_agent  # noqa: F401,E402
