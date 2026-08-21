from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    text: str
    score: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], source_text: str) -> "Span":
        start, end = int(value["start"]), int(value["end"])
        if not (0 <= start < end <= len(source_text)):
            raise ValueError(f"invalid span [{start}, {end}) for text of length {len(source_text)}")
        span_text = str(value.get("text", source_text[start:end]))
        if source_text[start:end] != span_text:
            raise ValueError(
                f"offset mismatch: text[{start}:{end}]={source_text[start:end]!r}, entity={span_text!r}"
            )
        return cls(start, end, str(value["label"]), span_text, _optional_float(value.get("score")))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "label": self.label,
        }
        if self.score is not None:
            value["score"] = self.score
        return value


@dataclass
class Example:
    id: str
    text: str
    entities: list[Span]
    split: str = "test"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Example":
        text = str(value["text"])
        return cls(
            id=str(value["id"]),
            text=text,
            entities=[Span.from_dict(entity, text) for entity in value.get("entities", [])],
            split=str(value.get("split", "test")),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "entities": [entity.to_dict() for entity in self.entities],
            "split": self.split,
            "metadata": self.metadata,
        }


def read_jsonl(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                example = Example.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if example.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {example.id!r}")
            seen.add(example.id)
            examples.append(example)
    return examples


def write_jsonl(path: str | Path, values: Iterable[dict[str, Any] | Example]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            record = value.to_dict() if isinstance(value, Example) else value
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark JSONL and character offsets.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    examples = read_jsonl(args.path)
    entities = sum(len(example.entities) for example in examples)
    print(f"valid: {len(examples)} examples, {entities} entities")


if __name__ == "__main__":
    main()
