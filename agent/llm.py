"""Centralized LLM client management and model configuration.

Provides:
  - YAML-based model config loading (``load_model_config`` / ``load_profile``)
  - Cached async/sync OpenAI-compatible clients
  - Unified chat completion for all API types (openai, azure, gemini, claude)
"""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


# ============================================================================
# Model config
# ============================================================================

def load_model_config(path: str) -> Dict[str, Any]:
    """Load the full model_config.yaml and return the ``profiles`` dict."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("profiles", {})


def load_defaults(path: str) -> Dict[str, Any]:
    """Load the ``defaults`` section from model_config.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return dict(raw.get("defaults", {}))


def load_profile(path: str, profile: str) -> Dict[str, Any]:
    """Load a single profile from model_config.yaml."""
    profiles = load_model_config(path)
    if profile not in profiles:
        available = ", ".join(profiles.keys()) or "(none)"
        raise KeyError(
            f"Profile {profile!r} not found in {path}. "
            f"Available profiles: {available}"
        )
    return dict(profiles[profile])


def list_profiles(path: str) -> List[str]:
    """Return the list of profile names in a model_config.yaml."""
    return list(load_model_config(path).keys())


# ============================================================================
# Response wrapper classes (duck-type compatible with OpenAI SDK)
# ============================================================================

class LLMFunction:
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class LLMToolCall:
    __slots__ = ("id", "type", "function", "thought_signature")

    def __init__(self, id: str, function: "LLMFunction", type: str = "function",
                 thought_signature: Optional[str] = None):
        self.id = id
        self.type = type
        self.function = function
        self.thought_signature = thought_signature


class LLMMessage:
    __slots__ = ("content", "tool_calls", "reasoning_content", "thinking_blocks")

    def __init__(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List["LLMToolCall"]] = None,
        reasoning_content: Optional[str] = None,
        thinking_blocks: Optional[list] = None,
    ):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.thinking_blocks = thinking_blocks


class LLMUsage:
    __slots__ = ("prompt_tokens", "completion_tokens", "thinking_tokens")

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, thinking_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.thinking_tokens = thinking_tokens


class _LLMChoice:
    __slots__ = ("message",)

    def __init__(self, message: "LLMMessage"):
        self.message = message


class LLMChatCompletion:
    __slots__ = ("choices", "usage")

    def __init__(self, message: "LLMMessage", usage: "LLMUsage"):
        self.choices = [_LLMChoice(message)]
        self.usage = usage


# ============================================================================
# OpenAI client pools
# ============================================================================

_async_cache: Dict[Tuple[str, str], AsyncOpenAI] = {}


def get_async_client(base_url: str, api_key: str = "EMPTY") -> AsyncOpenAI:
    base_url = base_url.rstrip("/")
    api_key = (api_key or "EMPTY").strip() or "EMPTY"
    key = (base_url, api_key)
    if key not in _async_cache:
        _async_cache[key] = AsyncOpenAI(
            base_url=base_url, api_key=api_key, max_retries=0,
        )
    return _async_cache[key]


_sync_cache: Dict[Tuple[str, str], OpenAI] = {}


def get_sync_client(base_url: str, api_key: str = "EMPTY") -> OpenAI:
    base_url = base_url.rstrip("/")
    api_key = (api_key or "EMPTY").strip() or "EMPTY"
    key = (base_url, api_key)
    if key not in _sync_cache:
        _sync_cache[key] = OpenAI(
            base_url=base_url, api_key=api_key, max_retries=0,
        )
    return _sync_cache[key]


# ============================================================================
# httpx client pool (for azure, gemini, claude)
# ============================================================================

_httpx_async_client: Optional[httpx.AsyncClient] = None
_httpx_sync_client: Optional[httpx.Client] = None


def _get_httpx_async() -> httpx.AsyncClient:
    global _httpx_async_client
    if _httpx_async_client is None:
        _httpx_async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0),
            follow_redirects=True,
        )
    return _httpx_async_client


def _get_httpx_sync() -> httpx.Client:
    global _httpx_sync_client
    if _httpx_sync_client is None:
        _httpx_sync_client = httpx.Client(
            timeout=httpx.Timeout(600.0),
            follow_redirects=True,
        )
    return _httpx_sync_client


# ============================================================================
# Cleanup
# ============================================================================

async def close_all_async_clients() -> None:
    global _httpx_async_client
    for client in _async_cache.values():
        await client.close()
    _async_cache.clear()
    if _httpx_async_client is not None:
        await _httpx_async_client.aclose()
        _httpx_async_client = None


def close_all_sync_clients() -> None:
    global _httpx_sync_client
    for client in _sync_cache.values():
        client.close()
    _sync_cache.clear()
    if _httpx_sync_client is not None:
        _httpx_sync_client.close()
        _httpx_sync_client = None


# ============================================================================
# Format translation: OpenAI → Gemini
# ============================================================================

def _messages_to_gemini(messages: list) -> Tuple[Optional[dict], list]:
    """Translate OpenAI-format messages to Gemini contents + systemInstruction."""
    tc_id_to_name: Dict[str, str] = {}
    for msg in messages:
        for tc in msg.get("tool_calls", []) or []:
            tc_id_to_name[tc["id"]] = tc["function"]["name"]

    system_instruction = None
    contents: list = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_instruction = {"parts": [{"text": msg.get("content", "")}]}
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts: list = []

        if role == "assistant":
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for tc in msg.get("tool_calls", []) or []:
                fn = tc["function"]
                args = fn["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {"raw": args}
                fc_part: dict = {"functionCall": {"name": fn["name"], "args": args}}
                if tc.get("thought_signature"):
                    fc_part["thoughtSignature"] = tc["thought_signature"]
                parts.append(fc_part)

        elif role == "tool":
            fn_name = msg.get("name") or tc_id_to_name.get(
                msg.get("tool_call_id", ""), "unknown"
            )
            parts.append({
                "functionResponse": {
                    "name": fn_name,
                    "response": {"result": msg.get("content", "")},
                }
            })
            gemini_role = "user"

        else:
            parts.append({"text": msg.get("content", "")})

        if not parts:
            continue

        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": gemini_role, "parts": parts})

    return system_instruction, contents


def _tools_to_gemini(tools: list) -> list:
    """Convert OpenAI tool definitions to Gemini function_declarations."""
    declarations = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            decl: dict = {"name": fn["name"]}
            if fn.get("description"):
                decl["description"] = fn["description"]
            if fn.get("parameters"):
                decl["parameters"] = fn["parameters"]
            declarations.append(decl)
    return [{"function_declarations": declarations}] if declarations else []


def _parse_gemini_response(data: dict) -> LLMChatCompletion:
    """Parse Gemini REST response into LLMChatCompletion."""
    candidates = data.get("candidates", [])
    if not candidates:
        return LLMChatCompletion(message=LLMMessage(content=""), usage=LLMUsage())

    parts = candidates[0].get("content", {}).get("parts", [])
    texts: list = []
    tool_calls: list = []
    reasoning: list = []

    for part in parts:
        if part.get("thought"):
            if "text" in part:
                reasoning.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append(LLMToolCall(
                id=f"gemini_{uuid.uuid4().hex[:8]}",
                function=LLMFunction(
                    name=fc["name"],
                    arguments=json.dumps(fc.get("args", {}), ensure_ascii=False),
                ),
                thought_signature=part.get("thoughtSignature"),
            ))
        elif "text" in part:
            texts.append(part["text"])

    usage_meta = data.get("usageMetadata", {})

    return LLMChatCompletion(
        message=LLMMessage(
            content="".join(texts) if texts else None,
            tool_calls=tool_calls or None,
            reasoning_content="".join(reasoning) if reasoning else None,
        ),
        usage=LLMUsage(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
            thinking_tokens=usage_meta.get("thoughtsTokenCount", 0),
        ),
    )


# ============================================================================
# Format translation: OpenAI → Claude
# ============================================================================

def _messages_to_claude(messages: list) -> Tuple[Optional[str], list]:
    """Translate OpenAI-format messages to Claude system + messages."""
    system_parts: list = []
    claude_msgs: list = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_parts.append(msg.get("content", ""))
            continue

        if role == "assistant":
            content_blocks: list = []
            if msg.get("thinking_blocks"):
                content_blocks.extend(msg["thinking_blocks"])
            elif msg.get("reasoning_content"):
                content_blocks.append({
                    "type": "thinking",
                    "thinking": msg["reasoning_content"],
                })
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []) or []:
                fn = tc["function"]
                input_data = fn["arguments"]
                if isinstance(input_data, str):
                    try:
                        input_data = json.loads(input_data)
                    except (json.JSONDecodeError, ValueError):
                        input_data = {"raw": input_data}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn["name"],
                    "input": input_data,
                })
            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})
            entry = {"role": "assistant", "content": content_blocks}

        elif role == "tool":
            entry = {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }],
            }

        else:
            entry = {"role": "user", "content": msg.get("content", "")}

        if claude_msgs and claude_msgs[-1]["role"] == entry["role"]:
            prev = claude_msgs[-1]
            if isinstance(prev["content"], str):
                prev["content"] = [{"type": "text", "text": prev["content"]}]
            if isinstance(entry["content"], str):
                entry_blocks = [{"type": "text", "text": entry["content"]}]
            else:
                entry_blocks = entry["content"]
            prev["content"].extend(entry_blocks)
        else:
            claude_msgs.append(entry)

    system = "\n\n".join(system_parts) if system_parts else None
    return system, claude_msgs


def _tools_to_claude(tools: list) -> list:
    """Convert OpenAI tool definitions to Claude format."""
    claude_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            ct: dict = {"name": fn["name"]}
            if fn.get("description"):
                ct["description"] = fn["description"]
            ct["input_schema"] = fn.get("parameters", {
                "type": "object", "properties": {},
            })
            claude_tools.append(ct)
    return claude_tools


def _parse_claude_response(data: dict) -> LLMChatCompletion:
    """Parse Claude Vertex rawPredict response into LLMChatCompletion."""
    content_blocks = data.get("content", [])
    texts: list = []
    tool_calls: list = []
    reasoning: list = []
    thinking_blocks: list = []

    for block in content_blocks:
        btype = block.get("type")
        if btype == "text":
            texts.append(block["text"])
        elif btype == "thinking":
            reasoning.append(block.get("thinking", ""))
            thinking_blocks.append({
                "type": "thinking",
                "thinking": block.get("thinking", ""),
                "signature": block.get("signature", ""),
            })
        elif btype == "tool_use":
            tool_calls.append(LLMToolCall(
                id=block["id"],
                function=LLMFunction(
                    name=block["name"],
                    arguments=json.dumps(
                        block.get("input", {}), ensure_ascii=False,
                    ),
                ),
            ))

    usage = data.get("usage", {})
    prompt_tokens = (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
    )

    return LLMChatCompletion(
        message=LLMMessage(
            content="".join(texts) if texts else None,
            tool_calls=tool_calls or None,
            reasoning_content="".join(reasoning) if reasoning else None,
            thinking_blocks=thinking_blocks or None,
        ),
        usage=LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=usage.get("output_tokens", 0),
        ),
    )


# ============================================================================
# Format translation: Azure response
# ============================================================================

def _parse_azure_response(data: dict) -> LLMChatCompletion:
    """Parse Azure OpenAI response into LLMChatCompletion."""
    if "choices" not in data or not data["choices"]:
        return LLMChatCompletion(message=LLMMessage(content=""), usage=LLMUsage())

    msg_data = data["choices"][0].get("message", {})

    tool_calls = None
    if msg_data.get("tool_calls"):
        tool_calls = [
            LLMToolCall(
                id=tc["id"],
                function=LLMFunction(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in msg_data["tool_calls"]
        ]

    usage_data = data.get("usage", {})
    details = usage_data.get("completion_tokens_details") or {}

    return LLMChatCompletion(
        message=LLMMessage(
            content=msg_data.get("content"),
            tool_calls=tool_calls,
            reasoning_content=msg_data.get("reasoning_content"),
        ),
        usage=LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            thinking_tokens=details.get("reasoning_tokens", 0) or 0,
        ),
    )


# ============================================================================
# Backend implementations (async)
# ============================================================================

async def _openai_async(cfg: dict, messages: list, **kwargs) -> Any:
    client = get_async_client(cfg["base_url"], cfg.get("api_key", "EMPTY"))
    call_kwargs: dict = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": kwargs.get("temperature", cfg.get("temperature", 0.7)),
        "top_p": kwargs.get("top_p", cfg.get("top_p", 0.95)),
        "max_tokens": kwargs.get("max_tokens", cfg.get("max_tokens", 8192)),
        "n": kwargs.get("n", 1),
        "stream": kwargs.get("stream", False),
    }
    if kwargs.get("extra_body"):
        call_kwargs["extra_body"] = kwargs["extra_body"]
    re_effort = kwargs.get("reasoning_effort") or cfg.get("reasoning_effort")
    if re_effort:
        call_kwargs["reasoning_effort"] = re_effort
    if kwargs.get("tools"):
        call_kwargs["tools"] = kwargs["tools"]
        call_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")
    return await client.chat.completions.create(**call_kwargs)


async def _azure_async(cfg: dict, messages: list, **kwargs) -> LLMChatCompletion:
    api_version = cfg.get("api_version", "2024-12-01-preview")
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "api-key": cfg.get("api_key", ""),
        "Content-Type": "application/json",
    }
    payload: dict = {
        "messages": messages,
        "temperature": kwargs.get("temperature", cfg.get("temperature", 1.0)),
        "max_completion_tokens": kwargs.get("max_tokens", cfg.get("max_tokens", 16384)),
    }
    re_effort = kwargs.get("reasoning_effort") or cfg.get("reasoning_effort")
    if re_effort:
        payload["reasoning_effort"] = re_effort
    if kwargs.get("tools"):
        payload["tools"] = kwargs["tools"]
        payload["tool_choice"] = kwargs.get("tool_choice", "auto")

    client = _get_httpx_async()
    resp = await client.post(
        url, params={"api-version": api_version},
        headers=headers, json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data or "Error" in data:
        err_detail = data.get("error") or data.get("Error", "")
        raise RuntimeError(f"Azure API error: {str(err_detail)[:500]}")
    return _parse_azure_response(data)


async def _gemini_async(cfg: dict, messages: list, **kwargs) -> LLMChatCompletion:
    headers = {
        "api-key": cfg.get("api_key", ""),
        "Content-Type": "application/json",
    }
    system_instruction, contents = _messages_to_gemini(messages)
    gen_config: dict = {
        "temperature": kwargs.get("temperature", cfg.get("temperature", 1.0)),
        "maxOutputTokens": kwargs.get("max_tokens", cfg.get("max_tokens", 16384)),
        "topP": kwargs.get("top_p", cfg.get("top_p", 0.95)),
    }
    thinking_level = cfg.get("thinking_level", "")
    if thinking_level:
        gen_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

    payload: dict = {
        "model": cfg.get("model", ""),
        "contents": contents,
        "generationConfig": gen_config,
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction
    if kwargs.get("tools"):
        payload["tools"] = _tools_to_gemini(kwargs["tools"])

    client = _get_httpx_async()
    resp = await client.post(cfg["base_url"], headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data or "Error" in data:
        err_detail = data.get("error") or data.get("Error", "")
        raise RuntimeError(f"Gemini API error: {str(err_detail)[:500]}")
    parsed = _parse_gemini_response(data)
    if (parsed.choices[0].message.content is None
            and not parsed.choices[0].message.tool_calls):
        finish = ""
        candidates = data.get("candidates", [])
        if candidates:
            finish = candidates[0].get("finishReason", "")
        logger.warning(
            "Gemini returned empty content and no tool_calls "
            "(finishReason=%s, usage=%s)",
            finish, data.get("usageMetadata", {}),
        )
    return parsed


def _claude_request_params(cfg: dict, messages: list, **kwargs):
    """Build URL, headers, payload for Claude API (standard / vertex / bedrock)."""
    base_url = cfg.get("base_url", "")
    is_bedrock = "bedrock" in base_url
    is_vertex = "vertex" in base_url or "rawPredict" in base_url or "generateContent" in base_url
    is_standard = not is_bedrock and not is_vertex

    if is_standard:
        url = f"{base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": cfg.get("api_key", ""),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        url = base_url
        headers = {
            "api-key": cfg.get("api_key", ""),
            "Content-Type": "application/json",
        }

    system, claude_messages = _messages_to_claude(messages)
    max_tokens = kwargs.get("max_tokens", cfg.get("max_tokens", 102400))
    thinking_budget = cfg.get("thinking_budget", 0)
    if thinking_budget and max_tokens <= thinking_budget:
        max_tokens = thinking_budget + 8192

    # Add cache_control to the last content block of the last user message
    for msg in reversed(claude_messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list) and content:
                content[-1]["cache_control"] = {"type": "ephemeral"}
            elif isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
                ]
            break

    payload: dict = {"max_tokens": max_tokens, "messages": claude_messages}
    if is_standard:
        payload["model"] = cfg.get("model", "")
    else:
        api_version = "bedrock-2023-05-31" if is_bedrock else "vertex-2023-10-16"
        payload["anthropic_version"] = api_version
    if system:
        if is_standard or is_vertex:
            payload["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]
        else:
            payload["system"] = system
    if thinking_budget:
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
    if kwargs.get("tools"):
        payload["tools"] = _tools_to_claude(kwargs["tools"])

    return url, headers, payload


async def _claude_async(cfg: dict, messages: list, **kwargs) -> LLMChatCompletion:
    url, headers, payload = _claude_request_params(cfg, messages, **kwargs)

    client = _get_httpx_async()
    resp = await client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data or "Error" in data:
        err_detail = data.get("error") or data.get("Error", "")
        raise RuntimeError(f"Claude API error: {str(err_detail)[:500]}")
    return _parse_claude_response(data)


# ============================================================================
# Backend implementations (sync)
# ============================================================================

def _openai_sync(cfg: dict, messages: list, **kwargs) -> Any:
    client = get_sync_client(cfg["base_url"], cfg.get("api_key", "EMPTY"))
    call_kwargs: dict = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": kwargs.get("temperature", cfg.get("temperature", 0.7)),
        "top_p": kwargs.get("top_p", cfg.get("top_p", 0.95)),
        "max_tokens": kwargs.get("max_tokens", cfg.get("max_tokens", 8192)),
        "n": kwargs.get("n", 1),
        "stream": kwargs.get("stream", False),
    }
    if kwargs.get("extra_body"):
        call_kwargs["extra_body"] = kwargs["extra_body"]
    re_effort = kwargs.get("reasoning_effort") or cfg.get("reasoning_effort")
    if re_effort:
        call_kwargs["reasoning_effort"] = re_effort
    if kwargs.get("tools"):
        call_kwargs["tools"] = kwargs["tools"]
        call_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")
    return client.chat.completions.create(**call_kwargs)


def _azure_sync(cfg: dict, messages: list, **kwargs) -> LLMChatCompletion:
    api_version = cfg.get("api_version", "2024-12-01-preview")
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "api-key": cfg.get("api_key", ""),
        "Content-Type": "application/json",
    }
    payload: dict = {
        "messages": messages,
        "temperature": kwargs.get("temperature", cfg.get("temperature", 1.0)),
        "max_completion_tokens": kwargs.get("max_tokens", cfg.get("max_tokens", 16384)),
    }
    re_effort = kwargs.get("reasoning_effort") or cfg.get("reasoning_effort")
    if re_effort:
        payload["reasoning_effort"] = re_effort
    if kwargs.get("tools"):
        payload["tools"] = kwargs["tools"]
        payload["tool_choice"] = kwargs.get("tool_choice", "auto")

    client = _get_httpx_sync()
    resp = client.post(
        url, params={"api-version": api_version},
        headers=headers, json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data or "Error" in data:
        err_detail = data.get("error") or data.get("Error", "")
        raise RuntimeError(f"Azure API error: {str(err_detail)[:500]}")
    return _parse_azure_response(data)


def _gemini_sync(cfg: dict, messages: list, **kwargs) -> LLMChatCompletion:
    headers = {
        "api-key": cfg.get("api_key", ""),
        "Content-Type": "application/json",
    }
    system_instruction, contents = _messages_to_gemini(messages)
    gen_config: dict = {
        "temperature": kwargs.get("temperature", cfg.get("temperature", 1.0)),
        "maxOutputTokens": kwargs.get("max_tokens", cfg.get("max_tokens", 16384)),
        "topP": kwargs.get("top_p", cfg.get("top_p", 0.95)),
    }
    thinking_level = cfg.get("thinking_level", "")
    if thinking_level:
        gen_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

    payload: dict = {
        "model": cfg.get("model", ""),
        "contents": contents,
        "generationConfig": gen_config,
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction
    if kwargs.get("tools"):
        payload["tools"] = _tools_to_gemini(kwargs["tools"])

    client = _get_httpx_sync()
    resp = client.post(cfg["base_url"], headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data or "Error" in data:
        err_detail = data.get("error") or data.get("Error", "")
        raise RuntimeError(f"Gemini API error: {str(err_detail)[:500]}")
    return _parse_gemini_response(data)


def _claude_sync(cfg: dict, messages: list, **kwargs) -> LLMChatCompletion:
    url, headers, payload = _claude_request_params(cfg, messages, **kwargs)

    client = _get_httpx_sync()
    resp = client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data or "Error" in data:
        err_detail = data.get("error") or data.get("Error", "")
        raise RuntimeError(f"Claude API error: {str(err_detail)[:500]}")
    return _parse_claude_response(data)


# ============================================================================
# Public API: unified chat completion
# ============================================================================

_ASYNC_DISPATCH = {
    "openai": _openai_async,
    "deployed": _openai_async,
    "azure": _azure_async,
    "gemini": _gemini_async,
    "claude": _claude_async,
}

_SYNC_DISPATCH = {
    "openai": _openai_sync,
    "deployed": _openai_sync,
    "azure": _azure_sync,
    "gemini": _gemini_sync,
    "claude": _claude_sync,
}


async def async_chat_completion(cfg: dict, messages: list, **kwargs):
    """Unified async chat completion — dispatches by ``cfg["api_type"]``.

    Args:
        cfg: Profile config dict (api_type, base_url, api_key, model, ...).
        messages: OpenAI-format message list.
        **kwargs: temperature, top_p, max_tokens, tools, tool_choice,
                  extra_body, reasoning_effort, n, stream.
    """
    api_type = cfg.get("api_type", "openai")
    handler = _ASYNC_DISPATCH.get(api_type)
    if handler is None:
        raise ValueError(f"Unsupported api_type: {api_type!r}")
    return await handler(cfg, messages, **kwargs)


def chat_completion(cfg: dict, messages: list, **kwargs):
    """Unified sync chat completion — dispatches by ``cfg["api_type"]``.

    Args:
        cfg: Profile config dict (api_type, base_url, api_key, model, ...).
        messages: OpenAI-format message list.
        **kwargs: temperature, top_p, max_tokens, tools, tool_choice,
                  extra_body, reasoning_effort, n, stream.
    """
    api_type = cfg.get("api_type", "openai")
    handler = _SYNC_DISPATCH.get(api_type)
    if handler is None:
        raise ValueError(f"Unsupported api_type: {api_type!r}")
    return handler(cfg, messages, **kwargs)
