#!/home/ubuntu/miniconda3/envs/cc/bin/python
"""Generate deterministic PD workload JSONL using the local tokenizer."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer


def make_prompt(tokenizer: Tokenizer, target: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    # Use ordinary printable words, then trim/pad the encoded sequence. The
    # API accepts token-id prompts, so the resulting request length is exact.
    words = [
        "analysis", "system", "request", "cache", "transfer", "latency",
        "prefill", "decode", "service", "benchmark", "token", "result",
    ]
    ids: list[int] = []
    while len(ids) < target:
        text = " ".join(rng.choice(words) for _ in range(32))
        ids.extend(tokenizer.encode(text, add_special_tokens=False).ids)
    return ids[:target]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ubuntu/zjj/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default="/home/ubuntu/zjj/code/workload.jsonl")
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--short-max", type=int, default=255)
    parser.add_argument("--medium-min", type=int, default=256)
    parser.add_argument("--medium-max", type=int, default=8192)
    parser.add_argument("--long-min", type=int, default=8193)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()
    if args.max_tokens > 100:
        raise SystemExit("--max-tokens must be <= 100")
    if args.requests < 3:
        raise SystemExit("--requests must be at least 3")

    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    rng = random.Random(args.seed)
    categories = ["short", "medium", "long"]
    lengths = {
        "short": max(1, min(args.short_max, 64)),
        "medium": max(args.medium_min, min(args.medium_max, 1024)),
        "long": max(args.long_min, min(args.long_min + 2048, 16384)),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(args.requests):
            category = categories[i % len(categories)]
            # Shuffle arrivals while keeping a balanced category mix.
            target = lengths[category] + rng.randrange(0, max(1, lengths[category] // 8))
            record = {
                "request_id": f"pd-{args.seed}-{i:05d}",
                "category": category,
                "seed": args.seed + i,
                "prompt": make_prompt(tokenizer, target, args.seed * 100000 + i),
                "max_tokens": min(args.max_tokens, 100),
            }
            out.write(json.dumps(record) + "\n")

    print(path)


if __name__ == "__main__":
    main()
