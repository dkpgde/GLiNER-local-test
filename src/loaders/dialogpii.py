from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from src.schema import Example, Span, write_jsonl


def _stable_split(key: str, dev_fraction: float, seed: int) -> str:
    value = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if value < dev_fraction else "test"


def _documents(source: Path, language: str, versions: set[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    language = language.upper()
    folders = {"original": "dialogs", "transcript": "transcripts"}
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                parts = PurePosixPath(name).parts
                for version in versions:
                    folder = folders[version]
                    if folder in parts and language in parts and name.endswith(".json"):
                        with archive.open(name) as handle:
                            yield name, json.load(handle)
                        break
        return

    for version in sorted(versions):
        folder = folders[version]
        candidates = list(source.glob(f"**/{folder}/{language}/*.json"))
        if not candidates and source.name == language:
            candidates = list(source.glob("*.json"))
        for path in sorted(candidates):
            yield path.as_posix(), json.loads(path.read_text(encoding="utf-8"))


def load_dialogpii(
    source: str | Path,
    language: str = "EN",
    versions: set[str] | None = None,
    dev_fraction: float = 0.2,
    seed: int = 42,
) -> list[Example]:
    versions = versions or {"original", "transcript"}
    examples: list[Example] = []
    for source_name, document in _documents(Path(source), language, versions):
        version = str(document.get("version", "original"))
        scenario = str(document["scenario"])
        for chat in document["dialogs"]:
            chat_number = chat["chat_number"]
            group = f"{language.upper()}:{scenario}:{chat_number}"
            split = _stable_split(group, dev_fraction, seed)
            for turn_index, turn in enumerate(chat["dialog"]["turns"]):
                text = str(turn["text"])
                entities = [
                    Span.from_dict(
                        {
                            "start": annotation["start"],
                            "end": annotation["end"],
                            "text": annotation["text"],
                            "label": annotation["type"],
                        },
                        text,
                    )
                    for annotation in turn.get("annotations", [])
                ]
                example_id = f"dialogpii_{version}_{language.lower()}_{scenario}_{chat_number:03d}_{turn_index:03d}"
                examples.append(
                    Example(
                        example_id,
                        text,
                        entities,
                        split,
                        {
                            "dataset": "dialogpii",
                            "language": language.upper(),
                            "scenario": scenario,
                            "version": version,
                            "chat_number": chat_number,
                            "turn_index": turn_index,
                            "speaker": turn.get("speaker"),
                            "source": source_name,
                        },
                    )
                )
    if not examples:
        raise ValueError(f"no DialogPII {language!r} documents found below {source}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official DialogPII files to benchmark JSONL.")
    parser.add_argument("source", type=Path, help="DialogPII.zip or its extracted directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="EN", help="Release language code, e.g. EN, DE, or SP")
    parser.add_argument("--version", choices=["original", "transcript", "both"], default="both")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        parser.error("--dev-fraction must be between 0 and 1")
    versions = {"original", "transcript"} if args.version == "both" else {args.version}
    examples = load_dialogpii(args.source, args.language, versions, args.dev_fraction, args.seed)
    write_jsonl(args.output, examples)
    counts = {split: sum(example.split == split for example in examples) for split in ("dev", "test")}
    print(f"wrote {len(examples)} turns to {args.output} ({counts['dev']} dev, {counts['test']} test)")


if __name__ == "__main__":
    main()
