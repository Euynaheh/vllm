#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request


def complete(url: str, model: str, prompt: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
            "seed": 20260820,
        }
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    assert body.get("choices"), body
    return body["choices"][0]["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()
    text = complete(args.url, args.model, args.prompt)
    print(repr(text))


if __name__ == "__main__":
    main()
