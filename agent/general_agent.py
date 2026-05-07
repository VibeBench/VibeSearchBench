"""General agent: OpenAI-compatible chat agent with context-length handling.

Features:
  1. _chat_with_retry: 400 BadRequestError not retried (deterministic error)
  2. model_context_limit param, max_context_tokens auto-capped
  3. _count_tokens: precise token counting via tokenizer
  4. _truncate_messages: truncation logic
  5. run_one: pre-request context check + truncation before force answer
  6. Single-agent and multi-agent modes
"""

import asyncio
import json
import logging
import os
import random
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI, BadRequestError

from . import BaseAgent, register, summarize_time_stats
from .llm import get_async_client, close_all_async_clients, async_chat_completion, LLMUsage
from .toolkit import ToolKit, BrowserToolKit

logger = logging.getLogger(__name__)

try:
    from json_repair import loads as _json_loads
except ImportError:
    logger.warning("json_repair not installed, using json.loads instead")
    _json_loads = json.loads  # type: ignore[assignment]

# ============================================================================
# Tool definitions (search / visit / python)
# ============================================================================

_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query string"},
        "topn":  {"type": "integer", "description": "Number of top results to display", "default": 10},
        "source": {"type": "string", "description": "Source to search within", "enum": ["web", "news"], "default": "web"},
    },
    "required": ["query"],
}

# builtin: search / open / find  — backed by BrowserPool (stateful page viewer)
TOOLS_BUILTIN = [
    {
        "type": "function",
        "function": {"name": "search", "description": "Searches for information related to query and displays topn results.", "parameters": _SEARCH_PARAMS},
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": (
                "Opens the link id from the page indicated by cursor starting at line number loc, "
                "showing num_lines lines. Valid link ids are displayed with the formatting: \u3010{id}\u2020.*\u3011. "
                "If cursor is not provided, the most recent page is implied. If id is a string, it is treated as a "
                "fully qualified URL associated with source. If loc is not provided, the viewport will be positioned "
                "at the beginning of the document or centered on the most relevant passage, if available. "
                "Use this function without id to scroll to a new location of an opened page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id":        {"description": "Link ID (number) or fully qualified URL (string)", "anyOf": [{"type": "integer"}, {"type": "string"}], "default": -1},
                    "cursor":    {"type": "integer", "description": "Page cursor indicator", "default": -1},
                    "loc":       {"type": "integer", "description": "Starting line number", "default": -1},
                    "num_lines": {"type": "integer", "description": "Number of lines to show", "default": -1},
                    "view_source": {"type": "boolean", "description": "Whether to view source", "default": False},
                    "source":    {"type": "string", "description": "Source associated with the URL", "enum": ["web", "news"], "default": "web"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Finds exact matches of pattern in the current page, or the page given by cursor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The exact pattern to search for"},
                    "cursor":  {"type": "integer", "description": "Page cursor to search in", "default": -1},
                },
                "required": ["pattern"],
            },
        },
    },
]

# custom: search / visit / python  — backed by ToolKit (Serper + LLM summarize + sandbox)
TOOLS_CUSTOM = [
    {
        "type": "function",
        "function": {"name": "search", "description": "Searches for information related to query and displays topn results.", "parameters": _SEARCH_PARAMS},
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit one or more webpages and return a summary of their content tailored to the specified goal. Returns separate summaries for each URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":  {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "A list of webpage URLs to visit."},
                    "goal": {"type": "string", "description": "The specific information to extract or focus on when summarizing the webpage content."},
                },
                "required": ["url", "goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scholar_search",
            "description": "Search Google Scholar for academic papers and publications. Returns titles, links, dates, sources, and snippets of scholarly articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query for Google Scholar"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "A utility that executes Python 3.11 code. Users must explicitly import any required libraries. The tool runs the provided Python code and returns both stdout and stderr. You should print results explicitly to ensure they appear in the returned output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python code to be executed."},
                },
                "required": ["code"],
            },
        },
    },
]

TOOL_SETS = {"builtin": TOOLS_BUILTIN, "custom": TOOLS_CUSTOM}

# Multi-agent: sub-agents run same tools/model in parallel
CREATE_SUB_AGENTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_sub_agents",
        "description": (
            "Create one or more sub-agents; each runs the given prompt with the same tools and model. "
            "Use this to split tasks and run them in parallel. Each sub-agent can search and visit links."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sub_agents": {
                    "type": "array",
                    "description": "List of sub-agents to run in parallel",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Task prompt for this sub-agent"},
                            "index": {"type": "integer", "description": "Unique index for this sub-agent"},
                        },
                        "required": ["prompt", "index"],
                    },
                },
            },
            "required": ["sub_agents"],
        },
    },
}

MULTI_AGENT_APPENDIX = """
3. Sub Agent: You can create one or more sub-agents via create_sub_agents. Each sub-agent runs your given prompt with the same search and link-reading tools. Split your task into sub-tasks and run them in parallel, then combine the results.
"""

COMPACT_SYSTEM = "You are a research summarizer. Your ONLY task is to produce a structured summary of a research conversation. You must NOT continue the conversation, answer questions, or provide recommendations. Output ONLY a factual summary."

COMPACT_PROMPT = """\
Please summarize the following research conversation. \
You MUST structure your summary by preserving EVERY user input/question as a section header, \
then list all relevant information gathered for that input.

For each user input, preserve:
- The exact user question/request (as section header)
- All key findings, facts, data points, and statistics related to that question
- Important URLs and sources discovered
- Search results and their relevant content
- Tool execution results and outputs
- Any conclusions or partial answers reached

IMPORTANT: This conversation may already contain a summary from an earlier compaction round. \
If you see a prior summary (e.g. "Here is a summary of my previous research"), you MUST \
incorporate ALL information from that earlier summary into your new summary. \
Do NOT discard earlier history — the final summary must cover the ENTIRE research from the very beginning.

Remove redundant information, intermediate reasoning steps, and repeated content. \
The summary must contain all information needed to continue the research task.

## Research Conversation
{conversation}

## IMPORTANT
Your task is to SUMMARIZE the conversation above, NOT to continue it. \
Do NOT answer questions or make recommendations. Output ONLY a structured factual summary.
"""

# ============================================================================
# Serialisation helpers
# ============================================================================

def _serialize_assistant(msg, *, require_reasoning: bool = False) -> dict:
    d: dict = {"role": "assistant"}
    if msg.content:
        d["content"] = msg.content
    if getattr(msg, "thinking_blocks", None):
        d["thinking_blocks"] = msg.thinking_blocks
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        d["reasoning_content"] = reasoning
    elif require_reasoning:
        d["reasoning_content"] = ""
    if msg.tool_calls:
        tcs = []
        for tc in msg.tool_calls:
            tc_dict = {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            if getattr(tc, "thought_signature", None):
                tc_dict["thought_signature"] = tc.thought_signature
            tcs.append(tc_dict)
        d["tool_calls"] = tcs
    return d


def _msg_for_api(serialized: dict) -> dict:
    api = {"role": serialized["role"]}
    api["content"] = serialized.get("content") or ""
    if "thinking_blocks" in serialized:
        api["thinking_blocks"] = serialized["thinking_blocks"]
    if "reasoning_content" in serialized:
        api["reasoning_content"] = serialized["reasoning_content"]
    if "tool_calls" in serialized:
        api["tool_calls"] = serialized["tool_calls"]
    return api


# ============================================================================
# Empty-response recovery: parse tool calls leaked into reasoning_content
# ============================================================================

_REASONING_TOOL_CALL_PARSERS: Dict[str, Any] = {}


def register_reasoning_parser(model_pattern: str, *, tags: tuple = ()):
    """Decorator: register a reasoning->tool-call parser for *model_pattern*."""
    def _wrap(fn):
        fn.tags = tags
        _REASONING_TOOL_CALL_PARSERS[model_pattern] = fn
        return fn
    return _wrap


@register_reasoning_parser("qwen", tags=("<tool_call>", "<function=", "<parameter="))
def _parse_qwen_tool_calls(reasoning: str) -> List[dict]:
    """Parse Qwen-3.5 style ``<tool_call>...</tool_call>`` blocks."""
    blocks = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>", reasoning, re.DOTALL,
    )
    if not blocks:
        return []
    results: List[dict] = []
    for block in blocks:
        fn_match = re.search(r"<function=(\w+)>(.*?)(?:</function>|$)", block, re.DOTALL)
        if not fn_match:
            continue
        fn_name = fn_match.group(1)
        body = fn_match.group(2)
        params: Dict[str, Any] = {}
        for pm in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", body, re.DOTALL,
        ):
            key = pm.group(1)
            val = pm.group(2)
            for caster in (int, float):
                try:
                    val = caster(val)  # type: ignore[assignment]
                    break
                except (ValueError, TypeError):
                    pass
            if isinstance(val, str) and val.startswith(("[", "{")):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    pass
            params[key] = val
        results.append({
            "id": f"recovered_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": fn_name, "arguments": json.dumps(params)},
        })
    return results


def _has_tool_call_tags(model_name: str, text: str) -> bool:
    """Check if *text* contains sentinel tags from the matched parser."""
    for pattern, parser in _REASONING_TOOL_CALL_PARSERS.items():
        if pattern in (model_name or "").lower():
            return any(tag in text for tag in getattr(parser, "tags", ()))
    return False


def _try_recover_tool_calls(
    model_name: str,
    reasoning: Optional[str],
    content: Optional[str] = None,
) -> List[dict]:
    """Attempt to recover tool calls from reasoning_content or content."""
    for source in (reasoning, content):
        if not source:
            continue
        for pattern, parser in _REASONING_TOOL_CALL_PARSERS.items():
            if pattern in (model_name or "").lower():
                try:
                    result = parser(source)
                    if result:
                        return result
                except Exception as exc:
                    logger.debug("reasoning parser %s failed: %s", pattern, exc)
    return []


# ============================================================================
# Agent
# ============================================================================


@register("general")
class GeneralAgent(BaseAgent):
    """OpenAI-compatible chat agent with context-length handling.

    Supports both single-agent and multi-agent modes with custom or builtin tool sets.
    """

    name = "Agent (General)"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        tool_set: str = "custom",
        developer_content: Optional[str] = None,
        sub_agent_developer_content: Optional[str] = None,
        max_rounds: int = 200,
        max_retries: int = 5,
        max_context_tokens: int = 120000,
        max_consecutive_errors: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        extra_body: Optional[dict] = None,
        # custom tool set options (search / visit / python)
        sandbox_url: Optional[str] = None,
        summarize_url: Optional[str] = None,
        summarize_model: Optional[str] = None,
        # builtin tool set options (search / open / find)
        browser_backend: str = "serper",
        search_url: str = "http://localhost:8000",
        # shared
        reasoning_effort: Optional[str] = None,
        multi_agent: bool = False,
        # api type (openai, azure, gemini, claude)
        api_type: str = "openai",
        api_version: Optional[str] = None,
        thinking_budget: int = 0,
        thinking_level: str = "",
        # context-length fix params
        model_context_limit: int = 262144,
        tokenizer_name: str = "Qwen/Qwen3.5-122B-A10B",
        **_kwargs,
    ):
        if tool_set not in TOOL_SETS:
            raise ValueError(f"tool_set must be one of {list(TOOL_SETS)}, got {tool_set!r}")
        self._tool_set = tool_set
        self._multi_agent = multi_agent
        self._base_url = base_url.rstrip("/")
        self._api_key = (api_key or "EMPTY").strip() or "EMPTY"
        self._model = model_name
        self._system_prompt = developer_content
        self._sub_agent_system_prompt = sub_agent_developer_content or developer_content
        self._max_rounds = max_rounds
        self._max_retries = max_retries
        self._max_tokens = max_tokens
        self._model_context_limit = model_context_limit
        # Cap max_context_tokens so that input + max_tokens <= model limit
        self._max_context_tokens = min(
            max_context_tokens,
            model_context_limit - self._max_tokens,
        )
        logger.info(
            "max_context_tokens capped to %d (model_limit=%d, max_tokens=%d, requested=%d)",
            self._max_context_tokens, model_context_limit, self._max_tokens, max_context_tokens,
        )
        self._max_consecutive_errors = max_consecutive_errors
        self._temperature = temperature
        self._top_p = top_p
        self._extra_body = dict(extra_body or {})
        if "deepseek" in (model_name or "").lower():
            self._extra_body["thinking"] = {"type": "enabled"}
            reasoning_effort = reasoning_effort or "max"
        elif "kimi" in (model_name or "").lower():
            self._extra_body["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": "max"}
        self._reasoning_effort = reasoning_effort
        self._api_type = api_type
        self._cfg: dict = {
            "api_type": api_type,
            "base_url": self._base_url,
            "api_key": self._api_key,
            "model": model_name,
        }
        if api_version:
            self._cfg["api_version"] = api_version
        if thinking_budget and api_type != "gemini":
            self._cfg["thinking_budget"] = thinking_budget
        if thinking_level:
            self._cfg["thinking_level"] = thinking_level
        if reasoning_effort:
            self._cfg["reasoning_effort"] = reasoning_effort
        # custom backend
        self._sandbox_url = sandbox_url
        self._serper_key = os.getenv("SERPER_API_KEY", "")
        self._summarize_url = summarize_url or base_url
        self._summarize_model = summarize_model or model_name
        # builtin backend
        self._browser_backend = browser_backend
        self._search_url = search_url
        # tokenizer for precise token counting
        self._tokenizer = None
        self._tokenizer_name = tokenizer_name
        self._tokenizer_failed = False
        # runtime
        self._client: Optional[AsyncOpenAI] = None
        self._toolkit: Optional[ToolKit] = None   # custom only
        self._pool = None  # builtin only

    # ---- token counting & truncation ----

    def _ensure_tokenizer(self):
        if self._tokenizer is None and not self._tokenizer_failed:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            except Exception as e:
                self._tokenizer_failed = True
                logger.warning("Failed to load tokenizer %s: %s. Using char-based estimation.", self._tokenizer_name, e)

    def _count_tokens(self, messages: list) -> int:
        """Token count using tokenizer + apply_chat_template, with fallback."""
        self._ensure_tokenizer()
        if self._tokenizer is not None:
            try:
                token_ids = self._tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                )
                return len(token_ids)
            except Exception:
                pass
        # Fallback: rough char-based estimation (~4 chars per token)
        num_tokens = 0
        for msg in messages:
            num_tokens += 4  # message format overhead
            for key, value in msg.items():
                if key == "tool_calls":
                    for tc in (value or []):
                        fn = tc.get("function", {})
                        num_tokens += len(fn.get("name", "")) // 4
                        num_tokens += len(fn.get("arguments", "")) // 4
                        num_tokens += 4
                elif isinstance(value, str):
                    num_tokens += len(value) // 4
        num_tokens += 2  # assistant priming
        return num_tokens

    def _truncate_messages(self, messages: list, max_input_tokens: int) -> list:
        """Truncate messages to fit within max_input_tokens.

        Strategy:
        1. Add messages one by one until exceeding the limit, then pop last.
        2. Find the nearest assistant-role boundary from the end.
        """
        truncated = []
        for msg in messages:
            truncated.append(msg)
            if self._count_tokens(truncated) > max_input_tokens:
                truncated.pop()
                break

        # Find nearest assistant boundary; keep at least system + user (first 2)
        min_keep = min(2, len(truncated))
        for i in range(len(truncated) - 1, min_keep - 1, -1):
            if truncated[i]["role"] == "assistant":
                truncated = truncated[:i + 1]
                break

        return truncated

    # ---- lifecycle ----

    async def setup(self):
        if self._api_type == "openai":
            self._client = get_async_client(self._base_url, self._api_key)
        if self._tool_set == "custom":
            if self._api_type == "openai" and self._summarize_url.rstrip("/") == self._base_url:
                summarize_client = self._client
            else:
                summarize_client = get_async_client(
                    self._summarize_url.rstrip("/"), "EMPTY",
                )
            self._toolkit = ToolKit(
                serper_api_key=self._serper_key,
                summarize_client=summarize_client,
                summarize_model=self._summarize_model,
                sandbox_url=self._sandbox_url,
            )
        else:
            try:
                from browser import BrowserPool
                self._pool = BrowserPool(
                    backend=self._browser_backend,
                    search_url=self._search_url,
                )
            except ImportError:
                raise ImportError(
                    "builtin tool set requires 'gpt_oss' and 'browser' modules. "
                    "Use --tool-set custom instead, or install the required packages."
                )

    def _make_toolkit_for_query(self, qid: Any):
        """Return the per-query tool backend."""
        if self._tool_set == "custom":
            return self._toolkit
        return BrowserToolKit(self._pool, qid)

    async def teardown(self):
        if self._pool is not None:
            await self._pool.close()
        await close_all_async_clients()

    # ---- chat completion ----

    async def _chat_with_retry(
        self,
        messages: list,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = "auto",
        max_tokens: Optional[int] = None,
    ) -> Tuple[Any, "LLMUsage"]:
        delay = 1.0
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = await async_chat_completion(
                    self._cfg, messages,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    max_tokens=max_tokens or self._max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    extra_body=self._extra_body or None,
                    reasoning_effort=self._reasoning_effort,
                    n=1,
                    stream=False,
                )
                raw_usage = resp.usage
                if isinstance(raw_usage, LLMUsage):
                    usage = raw_usage
                elif raw_usage:
                    thinking = 0
                    details = getattr(raw_usage, "completion_tokens_details", None)
                    if details:
                        thinking = getattr(details, "reasoning_tokens", 0) or 0
                    usage = LLMUsage(
                        prompt_tokens=raw_usage.prompt_tokens or 0,
                        completion_tokens=raw_usage.completion_tokens or 0,
                        thinking_tokens=thinking,
                    )
                else:
                    usage = LLMUsage()
                if not resp.choices:
                    raise RuntimeError(f"LLM returned response with no choices. Raw: {resp!r}")
                msg = resp.choices[0].message
                if msg.content is None and not getattr(msg, "tool_calls", None) and not getattr(msg, "reasoning_content", None) and not getattr(msg, "reasoning", None):
                    raise RuntimeError(
                        f"LLM returned empty response (no content, no tool_calls, no reasoning). "
                        f"Raw response type={type(resp).__name__}"
                    )
                return msg, usage
            except BadRequestError:
                # 400 is deterministic (e.g. context too long); retry is useless
                raise
            except Exception as e:
                if attempt == self._max_retries:
                    raise
                logger.warning(
                    "Chat completion attempt %d/%d failed (%s): %s",
                    attempt, self._max_retries, type(e).__name__, e,
                )
                await asyncio.sleep(delay * random.uniform(0.8, 1.2))
                delay *= 2.0

    # ---- agent loop (core primitives) ----

    async def _run_loop(
        self,
        trace: List[dict],
        api_msgs: List[dict],
        toolkit: "ToolKit",
        tools: list,
        qid: Any,
        max_rounds: int,
        time_stats: Dict[str, list],
        question: str = "",
        sub_agent_system_prompt_override: Optional[str] = None,
        token_usage: Optional[dict] = None,
        allow_compact: bool = True,
    ) -> Tuple[str, int, int]:
        """Core agent loop: LLM call -> tool execution -> repeat.

        Mutates *trace* and *api_msgs* in-place.
        If *token_usage* dict is provided, per-turn token counts are appended.

        Returns
        -------
        (termination, turns, last_prompt_tokens)
        """
        termination = "max_rounds"
        consecutive_errors = 0
        empty_retries = 0
        max_empty_retries = 5
        last_prompt_tokens = 0
        turns = 0

        for round_idx in range(max_rounds):
            trace_ckpt = len(trace)
            api_ckpt = len(api_msgs)

            # --- pre-request context length check: compact instead of break ---
            if last_prompt_tokens > self._max_context_tokens:
                if not allow_compact:
                    logger.warning("qid=%s: context too large but compact disabled, terminating", qid)
                    termination = "context_length"
                    break
                logger.info(
                    "qid=%s: prompt_tokens %d > %d, compacting context",
                    qid, last_prompt_tokens, self._max_context_tokens,
                )
                try:
                    turns += await self._compact_context(
                        trace, api_msgs, qid, question, time_stats, token_usage,
                    )
                except Exception as e:
                    logger.error("qid=%s: compact failed, terminating: %s", qid, e)
                    termination = "context_length"
                    break
                last_prompt_tokens = 0
                continue

            # --- LLM generation ---
            t_llm = time.time()
            try:
                resp_msg, usage = await self._chat_with_retry(
                    api_msgs, tools=tools,
                )
                prompt_tokens = usage.prompt_tokens
            except (BadRequestError, httpx.ReadTimeout) as e:
                if not allow_compact:
                    logger.warning("qid=%s: %s but compact disabled, terminating", qid, type(e).__name__)
                    termination = "context_length"
                    break
                logger.warning(
                    "qid=%s round=%d: %s, compacting: %s", qid, round_idx, type(e).__name__, e,
                )
                try:
                    turns += await self._compact_context(
                        trace, api_msgs, qid, question, time_stats, token_usage,
                    )
                except Exception as compact_err:
                    logger.error("qid=%s: compact failed, terminating: %s", qid, compact_err)
                    termination = "context_length"
                    break
                last_prompt_tokens = 0
                continue
            except Exception as e:
                consecutive_errors += 1
                logger.warning(
                    "qid=%s round=%d: generation failed (%d/%d) [%s]: %s",
                    qid, round_idx, consecutive_errors,
                    self._max_consecutive_errors, type(e).__name__, e,
                )
                if consecutive_errors >= self._max_consecutive_errors:
                    termination = "error_loop"
                    break
                continue

            time_stats["llm"].append(round(time.time() - t_llm, 3))
            last_prompt_tokens = prompt_tokens
            if token_usage is not None:
                token_usage["turns"].append({
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "thinking_tokens": usage.thinking_tokens,
                })
                token_usage["total_output_tokens"] += usage.completion_tokens
                token_usage["total_thinking_tokens"] += usage.thinking_tokens

            serialized = _serialize_assistant(
                resp_msg, require_reasoning=bool(self._extra_body.get("thinking")),
            )
            trace.append(serialized)
            api_msgs.append(_msg_for_api(serialized))
            turns += 1

            # --- No tool calls -> try recovery, then check for answer ---
            if not resp_msg.tool_calls:
                _ct = (resp_msg.content or "").strip()
                _reasoning = getattr(resp_msg, "reasoning_content", None) or getattr(resp_msg, "reasoning", None) or serialized.get("reasoning_content")
                recovered = _try_recover_tool_calls(self._model, _reasoning, _ct)
                if recovered:
                    logger.info(
                        "qid=%s round=%d: recovered %d tool call(s) from %s",
                        qid, round_idx, len(recovered),
                        "reasoning" if not _ct else "content",
                    )
                    serialized["tool_calls"] = recovered
                    serialized["content"] = None
                    trace[-1] = serialized
                    api_msgs[-1] = _msg_for_api(serialized)
                elif _ct and not _has_tool_call_tags(self._model, _ct):
                    termination = "answer"
                    consecutive_errors = 0
                    break
                else:
                    empty_retries += 1
                    if _ct:
                        logger.warning(
                            "qid=%s round=%d: tool call tags leaked into content, "
                            "rollback and retry (%d/%d)",
                            qid, round_idx, empty_retries, max_empty_retries,
                        )
                    else:
                        logger.warning(
                            "qid=%s round=%d: empty content (%d/%d), "
                            "has_reasoning=%s",
                            qid, round_idx, empty_retries, max_empty_retries,
                            bool(_reasoning),
                        )
                    if empty_retries >= max_empty_retries:
                        termination = "malformed_tool_call" if _ct else "empty_response"
                        break
                    del trace[trace_ckpt:]
                    del api_msgs[api_ckpt:]
                    turns -= 1
                    continue

            # --- Execute tool calls ---
            _tool_calls_to_run = serialized.get("tool_calls", [])
            if resp_msg.tool_calls:
                _tool_calls_to_run = None  # sentinel: use resp_msg directly

            rollback = False
            if _tool_calls_to_run is None:
                _tc_iter = resp_msg.tool_calls
            else:
                class _TC:
                    def __init__(self, d):
                        self.id = d["id"]
                        self.type = d["type"]
                        self.function = type("F", (), {
                            "name": d["function"]["name"],
                            "arguments": d["function"]["arguments"],
                        })()
                _tc_iter = [_TC(d) for d in _tool_calls_to_run]

            _trace_tc_by_id = {
                e["id"]: e for e in trace[-1].get("tool_calls", [])
            }

            parsed_calls = []
            for tc in _tc_iter:
                fn_name = tc.function.name
                raw_args = tc.function.arguments
                try:
                    args = _json_loads(raw_args)
                    if not isinstance(args, dict):
                        raise ValueError(f"parsed args is {type(args).__name__}, not dict")
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(
                        "qid=%s round=%d: bad args for '%s', rollback | "
                        "error=%s | raw=%r",
                        qid, round_idx, fn_name, e,
                        (raw_args or "")[:200],
                    )
                    rollback = True
                    break

                repaired = json.dumps(args, ensure_ascii=False)
                if repaired != raw_args and tc.id in _trace_tc_by_id:
                    _trace_tc_by_id[tc.id]["function"]["arguments"] = repaired
                parsed_calls.append((tc, fn_name, args))

            if not rollback and parsed_calls:
                async def _exec_tool(tc, fn_name, args):
                    t_tool = time.time()
                    try:
                        if fn_name == "create_sub_agents":
                            result = await self._run_create_sub_agents(
                                args, qid,
                                sub_prompt_override=sub_agent_system_prompt_override,
                            )
                        else:
                            result = await toolkit.call(fn_name, args)
                    except Exception as e:
                        result = f"Error executing '{fn_name}': {e}"
                    elapsed = round(time.time() - t_tool, 3)
                    return tc.id, fn_name, result, elapsed

                exec_results = await asyncio.gather(
                    *[_exec_tool(tc, fn, a) for tc, fn, a in parsed_calls]
                )
                for tc_id, fn_name, result, elapsed in exec_results:
                    time_stats.setdefault(fn_name, []).append(elapsed)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                    }
                    trace.append(tool_msg)
                    api_msgs.append(tool_msg)

            if rollback:
                del trace[trace_ckpt:]
                del api_msgs[api_ckpt:]
                consecutive_errors += 1
                if consecutive_errors >= self._max_consecutive_errors:
                    termination = "error_loop"
                    break
                continue

            consecutive_errors = 0

        return termination, turns, last_prompt_tokens

    async def _request_triple(
        self,
        trace: List[dict],
        api_msgs: List[dict],
        toolkit: "ToolKit",
        tools: list,
        qid: Any,
        triple_request_prompt: str,
        termination: str,
        time_stats: Dict[str, list],
        last_prompt_tokens: int,
        rounds_remaining: int,
        question: str = "",
        token_usage: Optional[dict] = None,
        sub_agent_system_prompt_override: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        """Append the triple-request prompt and run the agent loop.

        Returns (termination, additional_turns, last_prompt_tokens).
        """
        extra_turns = 0

        # Pre-compact if context is already near the limit — once we inject the
        # triple prompt we disable compact, so this is the last chance.
        if last_prompt_tokens > self._max_context_tokens * 0.85:
            logger.info(
                "qid=%s: pre-compacting before triple extraction (prompt_tokens=%d, limit=%d)",
                qid, last_prompt_tokens, self._max_context_tokens,
            )
            try:
                extra_turns += await self._compact_context(
                    trace, api_msgs, qid, question, time_stats, token_usage,
                )
            except Exception as e:
                logger.error("qid=%s: pre-compact before triple failed: %s", qid, e)
                return "context_length", extra_turns, last_prompt_tokens

        triple_msg = {"role": "user", "content": triple_request_prompt}
        trace.append(triple_msg)
        api_msgs.append(triple_msg)

        effective_rounds = max(rounds_remaining, 5)
        term, turns, lpt = await self._run_loop(
            trace, api_msgs, toolkit, None, qid,
            effective_rounds, time_stats,
            question=question,
            sub_agent_system_prompt_override=sub_agent_system_prompt_override,
            token_usage=token_usage,
            allow_compact=False,
        )
        extra_turns += turns
        if term != "answer":
            termination = term
        return termination, extra_turns, lpt

    async def _compact_context(
        self,
        trace: List[dict],
        api_msgs: List[dict],
        qid: Any,
        question: str,
        time_stats: Dict[str, list],
        token_usage: Optional[dict] = None,
    ) -> int:
        """Compact context using the agent's own model.

        Rebuilds *api_msgs* in-place to: [system?, question(user), summary(assistant)].
        Retries up to 3 times on failure; raises RuntimeError if all attempts fail.

        Returns 1 (the summary LLM call counts as one turn).
        """
        old_len = len(api_msgs)
        logger.info("qid=%s: compacting context (%d messages)", qid, old_len)

        system_msg = api_msgs[0] if api_msgs and api_msgs[0]["role"] == "system" else None

        conv_parts: list = []
        for m in api_msgs:
            if m["role"] == "system":
                continue
            role = m["role"]
            content = m.get("content") or ""
            if m.get("tool_calls"):
                tc_str = ", ".join(
                    tc["function"]["name"] for tc in m["tool_calls"]
                    if isinstance(tc, dict) and "function" in tc
                )
                content = f"{content} [tool_calls: {tc_str}]".strip()
            conv_parts.append(f"[{role}]: {content}")
        conversation_text = "\n".join(conv_parts)
        max_chars = min(self._max_context_tokens * 2, 300_000)
        if len(conversation_text) > max_chars:
            conversation_text = conversation_text[:max_chars]

        compact_msgs = [
            {"role": "system", "content": COMPACT_SYSTEM},
            {"role": "user", "content": COMPACT_PROMPT.format(conversation=conversation_text)},
        ]

        summary = ""
        usage = None
        t0 = time.time()
        for compact_attempt in range(1, 4):
            try:
                resp_msg, usage = await self._chat_with_retry(
                    compact_msgs, tools=None, tool_choice=None,
                )
                summary = (resp_msg.content or "").strip()
                if summary:
                    break
                logger.warning("qid=%s: compact attempt %d returned empty, retrying", qid, compact_attempt)
            except Exception as e:
                logger.warning("qid=%s: compact attempt %d failed: %s", qid, compact_attempt, e)
            if compact_attempt < 3:
                await asyncio.sleep(2.0 * compact_attempt)

        if not summary:
            raise RuntimeError(f"compact failed after 3 attempts for qid={qid}")

        time_stats["llm"].append(round(time.time() - t0, 3))
        if token_usage is not None and usage is not None:
            token_usage["turns"].append({
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "thinking_tokens": usage.thinking_tokens,
            })
            token_usage["total_output_tokens"] += usage.completion_tokens
            token_usage["total_thinking_tokens"] += usage.thinking_tokens

        if token_usage is not None:
            token_usage["compact_count"] = token_usage.get("compact_count", 0) + 1

        trace.append({"role": "system", "content": f"[Context compacted] {summary}"})

        api_msgs.clear()
        if system_msg:
            api_msgs.append(system_msg)
        api_msgs.append({"role": "user", "content": question})
        summary_msg: dict = {
            "role": "assistant",
            "content": f"Here is a summary of my previous research:\n\n{summary}",
        }
        if self._extra_body.get("thinking"):
            summary_msg["reasoning_content"] = ""
        api_msgs.append(summary_msg)
        api_msgs.append({
            "role": "user",
            "content": "Please continue your research based on the summary above.",
        })

        logger.info(
            "qid=%s: context compacted, api_msgs %d -> %d messages, summary %d chars",
            qid, old_len, len(api_msgs), len(summary),
        )
        return 1

    # ---- agent loop (public entry points) ----

    async def run_one(
        self,
        question: str,
        qid: Any,
        include_multi_agent_tool: Optional[bool] = None,
        system_prompt_override: Optional[str] = None,
        sub_agent_system_prompt_override: Optional[str] = None,
        triple_request_prompt: Optional[str] = None,
        **kwargs,
    ) -> dict:
        use_multi = (
            (include_multi_agent_tool if include_multi_agent_tool is not None else self._multi_agent)
            and self._multi_agent
        )
        tools = list(TOOL_SETS[self._tool_set])
        if use_multi:
            tools.append(CREATE_SUB_AGENTS_SCHEMA)
        toolkit = self._make_toolkit_for_query(qid)

        trace: List[dict] = []
        system_content = (system_prompt_override or self._system_prompt or "").strip()
        if system_content:
            trace.append({"role": "system", "content": system_content})
        trace.append({"role": "user", "content": question})
        api_msgs: List[dict] = list(trace)

        time_stats: Dict[str, list] = {"llm": []}
        token_usage: dict = {
            "total_output_tokens": 0,
            "total_thinking_tokens": 0,
            "compact_count": 0,
            "turns": [],
        }

        try:
            termination, turns, last_prompt_tokens = await self._run_loop(
                trace, api_msgs, toolkit, tools, qid,
                self._max_rounds, time_stats,
                question=question,
                sub_agent_system_prompt_override=sub_agent_system_prompt_override,
                token_usage=token_usage,
            )

            if triple_request_prompt:
                remaining = self._max_rounds - turns
                termination, extra, last_prompt_tokens = await self._request_triple(
                    trace, api_msgs, toolkit, tools, qid,
                    triple_request_prompt, termination, time_stats,
                    last_prompt_tokens, remaining,
                    question=question,
                    token_usage=token_usage,
                    sub_agent_system_prompt_override=sub_agent_system_prompt_override,
                )
                turns += extra

            self._finalize_token_usage(token_usage, last_prompt_tokens)
            return {
                "messages": trace,
                "termination": termination,
                "time_stats": summarize_time_stats(time_stats),
                "turns": turns,
                "prompt_tokens": last_prompt_tokens,
                "token_usage": token_usage,
            }

        finally:
            await toolkit.close()

    async def run_one_staged(
        self,
        initial_query: str,
        sub_queries: List[str],
        qid: Any,
        system_prompt_override: Optional[str] = None,
        triple_request_prompt: Optional[str] = None,
        sub_agent_system_prompt_override: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Staged mode: send initial_query then sub_queries sequentially.

        Conversation history is preserved across turns. The system prompt does
        NOT contain the triple output format — triples are requested explicitly
        via *triple_request_prompt* after all sub-queries are processed.
        """
        tools = list(TOOL_SETS[self._tool_set])
        toolkit = self._make_toolkit_for_query(qid)

        trace: List[dict] = []
        system_content = (system_prompt_override or self._system_prompt or "").strip()
        if system_content:
            trace.append({"role": "system", "content": system_content})
        trace.append({"role": "user", "content": initial_query})
        api_msgs: List[dict] = list(trace)

        time_stats: Dict[str, list] = {"llm": []}
        token_usage: dict = {
            "total_output_tokens": 0,
            "total_thinking_tokens": 0,
            "compact_count": 0,
            "turns": [],
        }
        rounds_used = 0
        total_turns = 0
        termination = "answer"
        last_prompt_tokens = 0

        try:
            # --- Phase 1: initial query ---
            remaining = self._max_rounds - rounds_used
            term, turns, last_prompt_tokens = await self._run_loop(
                trace, api_msgs, toolkit, tools, qid,
                remaining, time_stats,
                question=initial_query,
                sub_agent_system_prompt_override=sub_agent_system_prompt_override,
                token_usage=token_usage,
            )
            rounds_used += turns
            total_turns += turns
            if term == "error_loop":
                termination = term

            # --- Phase 2: sequential sub-queries ---
            if termination != "error_loop":
                for sq in sub_queries:
                    user_msg = {"role": "user", "content": sq}
                    trace.append(user_msg)
                    api_msgs.append(user_msg)

                    remaining = self._max_rounds - rounds_used
                    if remaining <= 0:
                        termination = "max_rounds"
                        break
                    term, turns, last_prompt_tokens = await self._run_loop(
                        trace, api_msgs, toolkit, tools, qid,
                        remaining, time_stats,
                        question=initial_query,
                        sub_agent_system_prompt_override=sub_agent_system_prompt_override,
                        token_usage=token_usage,
                    )
                    rounds_used += turns
                    total_turns += turns
                    if term == "error_loop":
                        termination = term
                        break

            # --- Phase 3: request triple output ---
            if triple_request_prompt:
                remaining = self._max_rounds - rounds_used
                termination, extra, last_prompt_tokens = await self._request_triple(
                    trace, api_msgs, toolkit, tools, qid,
                    triple_request_prompt, termination, time_stats,
                    last_prompt_tokens, remaining,
                    question=initial_query,
                    token_usage=token_usage,
                    sub_agent_system_prompt_override=sub_agent_system_prompt_override,
                )
                total_turns += extra

            self._finalize_token_usage(token_usage, last_prompt_tokens)
            return {
                "messages": trace,
                "termination": termination,
                "time_stats": summarize_time_stats(time_stats),
                "turns": total_turns,
                "prompt_tokens": last_prompt_tokens,
                "token_usage": token_usage,
            }

        finally:
            await toolkit.close()

    async def run_one_simulated(
        self,
        initial_query: str,
        user_persona: str,
        qid: Any,
        system_prompt_override: Optional[str] = None,
        triple_request_prompt: Optional[str] = None,
        sub_agent_system_prompt_override: Optional[str] = None,
        max_user_turns: int = 30,
        user_model_name: Optional[str] = None,
        user_model_url: Optional[str] = None,
        user_model_api_key: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Simulated mode: DR agent and user agent interact in alternating turns.

        Both agents maintain their own full context windows.  The DR agent
        uses tool calls to research; the user agent follows its persona to
        gradually disclose information needs.  When the user agent is
        satisfied it outputs ``[DONE]``, after which *triple_request_prompt*
        is injected to request KG triples.
        """
        from .prompts import USER_SIMULATOR_SYSTEM_PROMPT

        tools = list(TOOL_SETS[self._tool_set])
        toolkit = self._make_toolkit_for_query(qid)

        # --- DR agent context window ---
        trace: List[dict] = []
        system_content = (system_prompt_override or self._system_prompt or "").strip()
        if system_content:
            trace.append({"role": "system", "content": system_content})
        trace.append({"role": "user", "content": initial_query})
        api_msgs: List[dict] = list(trace)

        # --- User agent context window ---
        sim_system_prompt = USER_SIMULATOR_SYSTEM_PROMPT.format(
            user_persona=user_persona or "A curious researcher",
            initial_query=initial_query,
        )
        user_agent_msgs: List[dict] = [
            {"role": "system", "content": sim_system_prompt},
        ]

        # --- User agent LLM config ---
        if user_model_url:
            sim_cfg = {
                "api_type": "openai",
                "base_url": user_model_url,
                "api_key": user_model_api_key or "EMPTY",
                "model": user_model_name or self._model,
            }
        else:
            sim_cfg = dict(self._cfg)
            if user_model_name:
                sim_cfg["model"] = user_model_name
            if user_model_api_key:
                sim_cfg["api_key"] = user_model_api_key

        time_stats: Dict[str, list] = {"llm": []}
        token_usage: dict = {
            "total_output_tokens": 0,
            "total_thinking_tokens": 0,
            "compact_count": 0,
            "turns": [],
        }
        rounds_used = 0
        total_turns = 0
        termination = "answer"
        last_prompt_tokens = 0

        try:
            # --- Phase 1: DR agent responds to initial query ---
            remaining = self._max_rounds - rounds_used
            term, turns, last_prompt_tokens = await self._run_loop(
                trace, api_msgs, toolkit, tools, qid,
                remaining, time_stats,
                question=initial_query,
                sub_agent_system_prompt_override=sub_agent_system_prompt_override,
                token_usage=token_usage,
            )
            rounds_used += turns
            total_turns += turns
            if term == "error_loop":
                termination = term

            # --- Phase 2: DR agent <-> User agent alternating turns ---
            if termination != "error_loop":
                dr_response = self._extract_last_assistant_text(trace)

                for turn_num in range(1, max_user_turns + 1):
                    remaining = self._max_rounds - rounds_used
                    if remaining <= 0:
                        termination = "max_rounds"
                        break

                    # User agent receives DR response, generates follow-up
                    user_agent_msgs.append({"role": "user", "content": dr_response})
                    sim_response = await self._call_user_simulator(
                        user_agent_msgs, sim_cfg,
                    )
                    user_agent_msgs.append({"role": "assistant", "content": sim_response})

                    if "[DONE]" in sim_response:
                        logger.info("qid=%s: user agent signaled [DONE] at turn %d", qid, turn_num)
                        break

                    # DR agent receives user follow-up, researches and responds
                    user_msg = {"role": "user", "content": sim_response}
                    trace.append(user_msg)
                    api_msgs.append(user_msg)

                    term, turns, last_prompt_tokens = await self._run_loop(
                        trace, api_msgs, toolkit, tools, qid,
                        remaining, time_stats,
                        question=initial_query,
                        sub_agent_system_prompt_override=sub_agent_system_prompt_override,
                        token_usage=token_usage,
                    )
                    rounds_used += turns
                    total_turns += turns
                    if term == "error_loop":
                        termination = term
                        break

                    dr_response = self._extract_last_assistant_text(trace)

            # --- Phase 3: request triple output ---
            if triple_request_prompt:
                remaining = self._max_rounds - rounds_used
                termination, extra, last_prompt_tokens = await self._request_triple(
                    trace, api_msgs, toolkit, tools, qid,
                    triple_request_prompt, termination, time_stats,
                    last_prompt_tokens, remaining,
                    question=initial_query,
                    token_usage=token_usage,
                    sub_agent_system_prompt_override=sub_agent_system_prompt_override,
                )
                total_turns += extra

            self._finalize_token_usage(token_usage, last_prompt_tokens)
            return {
                "messages": trace,
                "termination": termination,
                "time_stats": summarize_time_stats(time_stats),
                "turns": total_turns,
                "prompt_tokens": last_prompt_tokens,
                "token_usage": token_usage,
            }

        finally:
            await toolkit.close()

    def _finalize_token_usage(self, token_usage: dict, last_prompt_tokens: int) -> None:
        """Compute trajectory_tokens and normalize total_output_tokens.

        trajectory_tokens = last input + last output (ignoring prior thinking).
        total_output_tokens = sum of all completion_tokens across turns.

        Gemini is the exception: its completion_tokens (candidatesTokenCount)
        excludes thinking, so thinking must be added to both metrics.
        """
        last_output = token_usage["turns"][-1]["output_tokens"] if token_usage["turns"] else 0
        token_usage["trajectory_tokens"] = last_prompt_tokens + last_output
        if self._api_type == "gemini":
            token_usage["trajectory_tokens"] += token_usage["total_thinking_tokens"]
            token_usage["total_output_tokens"] += token_usage["total_thinking_tokens"]

    @staticmethod
    def _extract_last_assistant_text(trace: List[dict]) -> str:
        """Extract the last assistant text response from trace."""
        for m in reversed(trace):
            if m.get("role") == "assistant" and m.get("content"):
                return m["content"]
        return ""

    async def _call_user_simulator(
        self,
        user_agent_msgs: List[dict],
        sim_cfg: dict,
    ) -> str:
        """Call the user simulator with its own maintained context window.

        *user_agent_msgs* is the simulator's full message history (system +
        alternating user/assistant turns).  The caller appends the DR agent's
        latest response as a ``user`` message before calling this method.
        """
        last_err = None
        for attempt in range(1, 4):
            try:
                resp = await async_chat_completion(
                    sim_cfg, user_agent_msgs,
                    temperature=0.7,
                    max_tokens=32768,
                )
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    raise RuntimeError("User simulator returned empty content")
                return content
            except Exception as e:
                last_err = e
                logger.warning("User simulator attempt %d/3 failed: %s", attempt, e)
                if attempt < 3:
                    await asyncio.sleep(1.0 * attempt)
        logger.error("User simulator gave up after 3 attempts: %s", last_err)
        return "[DONE]"

    async def _run_create_sub_agents(
        self, args: dict, qid: Any,
        sub_prompt_override: Optional[str] = None,
    ) -> str:
        sub_list = args.get("sub_agents") or []
        if not sub_list:
            return "[]"
        sub_prompt = sub_prompt_override or self._sub_agent_system_prompt
        tasks = [
            self.run_one(
                item.get("prompt", ""),
                f"{qid}_sub_{item.get('index', i)}",
                include_multi_agent_tool=False,
                system_prompt_override=sub_prompt,
            )
            for i, item in enumerate(sub_list)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for i, item in enumerate(sub_list):
            idx = item.get("index", i)
            prompt = item.get("prompt", "")
            if isinstance(results[i], Exception):
                out.append({"index": idx, "prompt": prompt, "response": str(results[i])})
            else:
                out.append({
                    "index": idx,
                    "prompt": prompt,
                    "response": self.extract_response(results[i].get("messages", [])),
                })
        return json.dumps(out, ensure_ascii=False)

    # ---- response extraction ----

    def extract_response(self, messages: List[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content") or ""
            if isinstance(content, str) and content.strip():
                return content.strip()
        return ""
