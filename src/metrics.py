from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from src.schema import Example, Span


def _compatible(gold: Span, predicted: Span, overlap: bool) -> bool:
    if gold.label != predicted.label:
        return False
    if overlap:
        return max(gold.start, predicted.start) < min(gold.end, predicted.end)
    return gold.start == predicted.start and gold.end == predicted.end


def match_spans(gold: Sequence[Span], predicted: Sequence[Span], overlap: bool = False) -> tuple[set[int], set[int]]:
    """Return maximum-cardinality label-aware span matches."""
    edges = [[index for index, item in enumerate(gold) if _compatible(item, candidate, overlap)] for candidate in predicted]
    gold_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for gold_index in edges[prediction_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            if gold_index not in gold_to_prediction or augment(gold_to_prediction[gold_index], visited):
                gold_to_prediction[gold_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predicted)):
        augment(prediction_index, set())
    return set(gold_to_prediction), set(gold_to_prediction.values())


def _score(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}


def span_metrics(
    examples: Sequence[Example],
    predictions: Mapping[str, Sequence[Span]],
    labels: Iterable[str] | None = None,
    overlap: bool = False,
) -> dict:
    selected = set(labels) if labels is not None else None
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for example in examples:
        gold = [span for span in example.entities if selected is None or span.label in selected]
        predicted = [span for span in predictions.get(example.id, []) if selected is None or span.label in selected]
        gold_matches, prediction_matches = match_spans(gold, predicted, overlap)
        for index, span in enumerate(gold):
            counts[span.label][0 if index in gold_matches else 2] += 1
        for index, span in enumerate(predicted):
            if index not in prediction_matches:
                counts[span.label][1] += 1
    if selected is not None:
        for label in selected:
            counts[label]
    per_label = {label: _score(*values) for label, values in sorted(counts.items())}
    totals = [sum(values[index] for values in counts.values()) for index in range(3)]
    micro = _score(*totals)
    macro_f1 = sum(value["f1"] for value in per_label.values()) / len(per_label) if per_label else 0.0
    return {"micro": micro, "macro_f1": macro_f1, "per_label": per_label}


def evaluate(
    examples: Sequence[Example],
    predictions: Mapping[str, Sequence[Span]],
    groups: Mapping[str, Sequence[str]] | None = None,
    privacy_labels: Sequence[str] | None = None,
) -> dict:
    result = {
        "exact": span_metrics(examples, predictions),
        "relaxed": span_metrics(examples, predictions, overlap=True),
        "groups": {},
    }
    for name, labels in (groups or {}).items():
        result["groups"][name] = {
            "exact": span_metrics(examples, predictions, labels),
            "relaxed": span_metrics(examples, predictions, labels, overlap=True),
        }
    if privacy_labels:
        exact = span_metrics(examples, predictions, privacy_labels)
        relaxed = span_metrics(examples, predictions, privacy_labels, overlap=True)
        support = exact["micro"]["support"]
        result["privacy"] = {
            "exact": exact,
            "relaxed": relaxed,
            "false_negatives_per_1000": exact["micro"]["fn"] / support * 1000 if support else 0.0,
        }
    return result
