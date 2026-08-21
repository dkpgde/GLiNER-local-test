from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.metrics import match_spans
from src.schema import Span


def build_error_rows(prediction_path: Path) -> list[dict]:
    rows: list[dict] = []
    with prediction_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record["text"]
            gold = [Span.from_dict(value, text) for value in record["gold_entities"]]
            predicted = [Span.from_dict(value, text) for value in record["predicted_entities"]]
            gold_matches, prediction_matches = match_spans(gold, predicted)
            for index, span in enumerate(gold):
                if index not in gold_matches:
                    rows.append(_row(record["id"], "false_negative", span))
            for index, span in enumerate(predicted):
                if index not in prediction_matches:
                    rows.append(_row(record["id"], "false_positive", span))
    return rows


def _row(example_id: str, error_type: str, span: Span) -> dict:
    return {
        "example_id": example_id,
        "error_type": error_type,
        "label": span.label,
        "start": span.start,
        "end": span.end,
        "text": span.text,
        "score": "" if span.score is None else span.score,
        "category": "",
        "material_impact": "",
        "notes": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced GLiNER error-review CSV.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/error_analysis.csv"))
    parser.add_argument("--per-type", type=int, default=50)
    args = parser.parse_args()
    rows = build_error_rows(args.predictions)
    selected = []
    for error_type in ("false_negative", "false_positive"):
        candidates = [row for row in rows if row["error_type"] == error_type]
        selected.extend(candidates[: args.per_type])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_row("", "", Span(0, 1, "", "x")).keys()))
        writer.writeheader()
        writer.writerows(selected)
    print(f"wrote {len(selected)} review rows to {args.output}")


if __name__ == "__main__":
    main()
