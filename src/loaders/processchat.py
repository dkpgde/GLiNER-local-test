from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator

from src.schema import Example, write_jsonl


TURN_RE = re.compile(r"<\|(user|assistant)\|>\s*\n(.*?)(?=\n<\|(?:user|assistant)\|>|\Z)", re.DOTALL)


def _source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    # A cloned ProcessChat repo also contains two ablation datasets. Prefer the
    # complete official `data` split unless the caller names another directory.
    complete = source / "data"
    return sorted((complete if complete.is_dir() else source).rglob("*.jsonl"))


def _candidates(source: Path) -> Iterator[dict[str, Any]]:
    for path in _source_files(source):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                turns = [(speaker, text.strip()) for speaker, text in TURN_RE.findall(record["input"]) if text.strip()]
                last_user = next((text for speaker, text in reversed(turns) if speaker == "user"), None)
                for speaker, text in (("user", last_user), ("assistant", str(record.get("output", "")).strip())):
                    if text:
                        yield {
                            "speaker": speaker,
                            "text": text,
                            "source": path.as_posix(),
                            "source_line": line_number,
                        }


def _bucket(text: str) -> str:
    words = len(text.split())
    return "short" if words < 8 else "medium" if words < 25 else "long"


def _sample_stratified(values: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for value in values:
        strata[(value["speaker"], _bucket(value["text"]))].append(value)
    for items in strata.values():
        shuffled = list(items)
        rng.shuffle(shuffled)
        items.clear()
        items.extend(shuffled)
    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(selected) < size and any(strata.values()):
        rng.shuffle(keys)
        for key in keys:
            if strata[key] and len(selected) < size:
                selected.append(strata[key].popleft())
    return selected


def prepare_processchat(
    source: str | Path,
    sample_size: int = 250,
    dev_fraction: float = 0.2,
    seed: int = 42,
) -> list[Example]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for value in _candidates(Path(source)):
        normalized = " ".join(value["text"].split()).casefold()
        unique.setdefault((value["speaker"], normalized), value)
    sampled = _sample_stratified(list(unique.values()), min(sample_size, len(unique)), seed)
    examples: list[Example] = []
    for value in sampled:
        digest = hashlib.sha1(f"{value['speaker']}\0{value['text']}".encode()).hexdigest()
        fraction = int(hashlib.sha1(f"{seed}:{digest}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        examples.append(
            Example(
                id=f"processchat_{digest[:12]}",
                text=value["text"],
                entities=[],
                split="dev" if fraction < dev_fraction else "test",
                metadata={
                    "dataset": "processchat",
                    "speaker": value["speaker"],
                    "length_bucket": _bucket(value["text"]),
                    "source": value["source"],
                    "source_line": value["source_line"],
                },
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a ProcessChat manual-annotation template.")
    parser.add_argument("source", type=Path, help="Official JSONL file or directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_size < 1 or not 0 < args.dev_fraction < 1:
        parser.error("sample size must be positive and dev fraction must be between 0 and 1")
    examples = prepare_processchat(args.source, args.sample_size, args.dev_fraction, args.seed)
    write_jsonl(args.output, examples)
    print(f"wrote {len(examples)} annotation candidates to {args.output}")


if __name__ == "__main__":
    main()
