#!/usr/bin/env python3
from __future__ import annotations

import argparse

from smoke_http import complete


PROMPTS = (
    "The capital of France is",
    "Write one sentence explaining tensor parallelism:",
    "1, 1, 2, 3, 5, 8,",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mega-url", required=True)
    parser.add_argument("--native-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    mismatches = []
    for prompt in PROMPTS:
        mega = complete(args.mega_url, args.model, prompt)
        native = complete(args.native_url, args.model, prompt)
        print(f"prompt={prompt!r}\n  mega={mega!r}\n  native={native!r}")
        if mega != native:
            mismatches.append(prompt)
    if mismatches:
        raise SystemExit(f"backend output mismatch for {len(mismatches)} prompts")


if __name__ == "__main__":
    main()
