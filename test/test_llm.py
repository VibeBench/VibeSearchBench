#!/usr/bin/env python3
"""Test all LLM profiles in model_config.yaml by asking each model to introduce itself."""

import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.llm import load_model_config, get_sync_client

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "model_config.yaml")

PROMPT = "请用一两句话介绍你自己的模型型号和版本。"


def _print_openai_usage(usage, label=""):
    """Print OpenAI usage with cached/reasoning token details."""
    if not usage:
        return
    prefix = f"  [{label}] " if label else "  "
    parts = [f"prompt={usage.prompt_tokens}", f"completion={usage.completion_tokens}"]
    pd = getattr(usage, "prompt_tokens_details", None)
    if pd:
        cached = getattr(pd, "cached_tokens", None)
        if cached is not None:
            parts.append(f"cached={cached}")
    cd = getattr(usage, "completion_tokens_details", None)
    if cd:
        reasoning = getattr(cd, "reasoning_tokens", None)
        if reasoning is not None:
            parts.append(f"reasoning={reasoning}")
    print(f"{prefix}Tokens: {', '.join(parts)}")


def test_openai(name, cfg):
    """Test an OpenAI-compatible endpoint."""
    client = get_sync_client(cfg["base_url"], cfg.get("api_key", "EMPTY"))
    extra = {}
    if cfg.get("reasoning_effort"):
        extra["reasoning_effort"] = cfg["reasoning_effort"]
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": PROMPT}],
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 1024),
        **extra,
    )
    text = (resp.choices[0].message.content or "").strip()
    print(f"  [OK] Response: {text}")
    _print_openai_usage(resp.usage)


def test_azure(name, cfg):
    """Test an Azure-style endpoint (api-key header, api-version query param)."""
    api_version = cfg.get("api_version", "2024-12-01-preview")
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "api-key": cfg.get("api_key", ""),
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": cfg.get("temperature", 1.0),
        "max_completion_tokens": cfg.get("max_tokens", 16384),
    }
    if cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg["reasoning_effort"]
    client = httpx.Client(timeout=httpx.Timeout(60.0), follow_redirects=True)
    try:
        resp = client.post(url, params={"api-version": api_version}, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data:
            raise ValueError(f"Unexpected response: {data}")
        text = data["choices"][0]["message"]["content"].strip()
        print(f"  [OK] Response: {text}")

        usage = data.get("usage", {})
        if usage:
            print(f"  Tokens: prompt={usage.get('prompt_tokens', '?')}, "
                  f"completion={usage.get('completion_tokens', '?')}")
    finally:
        client.close()


def test_gemini(name, cfg):
    """Test a Gemini REST endpoint (:generateContent)."""
    headers = {
        "api-key": cfg.get("api_key", ""),
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.get("model", ""),
        "contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
        "generationConfig": {
            "temperature": cfg.get("temperature", 1.0),
            "maxOutputTokens": cfg.get("max_tokens", 16384),
            "topP": cfg.get("top_p", 0.95),
        },
    }
    client = httpx.Client(timeout=httpx.Timeout(60.0), headers=headers, follow_redirects=True)
    try:
        resp = client.post(cfg["base_url"], json=payload)
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"Empty candidates: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p["text"] for p in parts if "text" in p and p.get("thought") is None]
        if not texts:
            raise ValueError("No text parts in response")

        print(f"  [OK] Response: {''.join(texts)}")

        usage = data.get("usageMetadata", {})
        if usage:
            print(f"  Tokens: prompt={usage.get('promptTokenCount', '?')}, "
                  f"completion={usage.get('candidatesTokenCount', '?')}")
    finally:
        client.close()


def _claude_call(url, headers, payload):
    """Send one Claude request and return parsed data."""
    client = httpx.Client(timeout=httpx.Timeout(120.0), follow_redirects=True)
    try:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()
    finally:
        client.close()


def _print_claude_usage(data, turn_label=""):
    """Print token usage including cache stats."""
    usage = data.get("usage", {})
    if not usage:
        return
    parts = [
        f"input={usage.get('input_tokens', '?')}",
        f"output={usage.get('output_tokens', '?')}",
    ]
    cache_creation = usage.get("cache_creation_input_tokens")
    cache_read = usage.get("cache_read_input_tokens")
    if cache_creation is not None:
        parts.append(f"cache_creation={cache_creation}")
    if cache_read is not None:
        parts.append(f"cache_read={cache_read}")
    prefix = f"  [{turn_label}] " if turn_label else "  "
    print(f"{prefix}Tokens: {', '.join(parts)}")


def test_claude(name, cfg):
    """Test Claude with multi-turn cache: turn 1 warms cache, turn 2 should hit it."""
    from agent.llm import _claude_request_params

    # --- Turn 1: initial question ---
    messages_t1 = [{"role": "user", "content": PROMPT}]
    url, headers, payload = _claude_request_params(cfg, messages_t1, max_tokens=1024)
    print("  Turn 1: sending initial request...")
    data1 = _claude_call(url, headers, payload)

    blocks1 = data1.get("content", [])
    texts1 = [b["text"] for b in blocks1 if b.get("type") == "text"]
    if not texts1:
        raise ValueError(f"No text blocks in turn-1 response: {data1}")
    reply1 = "".join(texts1)
    print(f"  [OK] Turn 1 response: {reply1[:120]}")
    _print_claude_usage(data1, "Turn 1")

    # --- Turn 2: follow-up (should hit cache on turn-1 tokens) ---
    # Build assistant message in OpenAI format so _messages_to_claude handles it correctly
    thinking_blocks = [b for b in blocks1 if b.get("type") == "thinking"]
    assistant_msg = {"role": "assistant", "content": reply1}
    if thinking_blocks:
        assistant_msg["thinking_blocks"] = thinking_blocks

    messages_t2 = [
        {"role": "user", "content": PROMPT},
        assistant_msg,
        {"role": "user", "content": "请再详细说说你的参数量和训练数据截止日期。"},
    ]
    url2, headers2, payload2 = _claude_request_params(cfg, messages_t2, max_tokens=1024)
    print("  Turn 2: sending follow-up (expect cache hit)...")
    data2 = _claude_call(url2, headers2, payload2)

    blocks2 = data2.get("content", [])
    texts2 = [b["text"] for b in blocks2 if b.get("type") == "text"]
    if not texts2:
        raise ValueError(f"No text blocks in turn-2 response: {data2}")
    print(f"  [OK] Turn 2 response: {''.join(texts2)[:120]}")
    _print_claude_usage(data2, "Turn 2")


LONG_SYSTEM_PROMPT = (
    "你是一位百科全书式的助手。以下是你需要掌握的背景知识：\n\n"
    + "\n".join(
        f"第{i}条：人工智能（Artificial Intelligence）是计算机科学的一个分支，"
        f"它试图理解智能的本质并生产出一种新的能以与人类智能相似的方式做出反应的智能机器。"
        f"研究领域包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"
        f"自从1956年达特茅斯会议以来，AI经历了多次发展浪潮，"
        f"从符号主义到连接主义再到深度学习，每一次变革都推动了技术的进步。"
        f"当前大语言模型（LLM）的突破代表了AI领域的最新进展。（编号={i}）"
        for i in range(1, 201)
    )
)


def test_openai_cache(name, cfg):
    """Send the same long prompt twice to test prompt caching."""
    client = get_sync_client(cfg["base_url"], cfg.get("api_key", "EMPTY"))
    extra = {}
    if cfg.get("reasoning_effort"):
        extra["reasoning_effort"] = cfg["reasoning_effort"]

    messages = [
        {"role": "system", "content": LONG_SYSTEM_PROMPT},
        {"role": "user", "content": "请用一句话总结上面的背景知识。"},
    ]

    print("  Cache test: request 1 (warm up)...")
    resp1 = client.chat.completions.create(
        model=cfg["model"], messages=messages,
        temperature=cfg.get("temperature", 0.7),
        max_tokens=512, **extra,
    )
    text1 = (resp1.choices[0].message.content or "").strip()
    print(f"  [OK] Response 1: {text1[:120]}")
    _print_openai_usage(resp1.usage, "Req 1")

    print("  Cache test: request 2 (expect cache hit)...")
    resp2 = client.chat.completions.create(
        model=cfg["model"], messages=messages,
        temperature=cfg.get("temperature", 0.7),
        max_tokens=512, **extra,
    )
    text2 = (resp2.choices[0].message.content or "").strip()
    print(f"  [OK] Response 2: {text2[:120]}")
    _print_openai_usage(resp2.usage, "Req 2")


def test_profile(name, cfg, cache_test=False):
    api_type = cfg.get("api_type", "openai")
    model = cfg.get("model", "")
    base_url = cfg.get("base_url", "")

    print(f"\n{'=' * 60}")
    print(f"Profile: {name}")
    print(f"  Model:    {model}")
    print(f"  Base URL: {base_url}")
    print(f"  Type:     {api_type}")
    print(f"{'=' * 60}")

    if not model or not base_url:
        print("  [SKIP] model or base_url not set")
        return False

    try:
        if cache_test:
            if api_type in ("openai", "deployed"):
                test_openai_cache(name, cfg)
            else:
                print(f"  [SKIP] cache test not implemented for api_type={api_type}")
                return True
        elif api_type == "gemini":
            test_gemini(name, cfg)
        elif api_type == "azure":
            test_azure(name, cfg)
        elif api_type == "claude":
            test_claude(name, cfg)
        else:
            test_openai(name, cfg)
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser(description="Test all LLM profiles in model_config.yaml")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to model_config.yaml")
    p.add_argument("--profile", default=None, help="Test a single profile (default: all)")
    p.add_argument("--cache-test", action="store_true", help="Test prompt caching with long prompt x2")
    args = p.parse_args()

    profiles = load_model_config(args.config)
    if not profiles:
        print(f"No profiles found in {args.config}")
        sys.exit(1)

    if args.profile:
        if args.profile not in profiles:
            print(f"Profile {args.profile!r} not found. Available: {list(profiles.keys())}")
            sys.exit(1)
        profiles = {args.profile: profiles[args.profile]}

    print(f"Testing {len(profiles)} profile(s) from {args.config}")

    results = {}
    for name, cfg in profiles.items():
        results[name] = test_profile(name, cfg, cache_test=args.cache_test)

    print(f"\n{'=' * 60}")
    print("Summary:")
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"{'=' * 60}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
