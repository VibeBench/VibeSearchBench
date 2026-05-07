#!/usr/bin/env python3
"""Quick smoke test for vLLM endpoint connectivity."""

import argparse
import sys

from openai import OpenAI


def main():
    p = argparse.ArgumentParser(description="Test vLLM endpoint connectivity")
    p.add_argument("--vllm-url", default="http://localhost:80/v1", help="Base URL, e.g. http://host/v1")
    p.add_argument("--model", default="glm-5.1", help="Model name")
    p.add_argument("--api-key", default="EMPTY", help="API key (default: EMPTY)")
    args = p.parse_args()

    client = OpenAI(base_url=args.vllm_url.rstrip("/"), api_key=args.api_key)

    # 1. List models
    print(f">>> Testing {args.vllm_url} with model={args.model}")
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"[OK] /models returned {len(available)} model(s): {available}")
    except Exception as e:
        print(f"[WARN] /models failed: {e}")

    # 2. Chat completion
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=32000,
            temperature=0.6,
        )
        text = resp.choices[0].message.content.strip()
        tokens = resp.usage
        print(f"[OK] Chat completion succeeded")
        print(f"     Response: {text}")
        print(f"     Tokens: prompt={tokens.prompt_tokens}, completion={tokens.completion_tokens}")
    except Exception as e:
        print(f"[FAIL] Chat completion failed: {e}")
        sys.exit(1)

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
