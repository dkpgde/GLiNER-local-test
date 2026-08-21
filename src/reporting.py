from __future__ import annotations

import csv
from pathlib import Path


def write_comparison(summary_path: Path, output_path: Path, decision_criteria: dict | None = None) -> None:
    with summary_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    headers = [
        "Model",
        "Process NER F1",
        "Technology F1",
        "Improvement Signal F1",
        "PII Recall",
        "PII F1",
        "Median Latency",
        "p95 Latency",
        "Model RAM",
    ]
    lines = ["# Final model comparison", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * 8) + "|"]
    for row in rows:
        lines.append(
            "| {model} | {process_f1} | {technology_f1} | {improvement_signal_f1} | {pii_recall} | "
            "{pii_f1} | {median_latency_ms} ms | {p95_latency_ms} ms | {model_ram_mb} MB |".format(**row)
        )
    by_model = {row["model"]: row for row in rows}
    if "bi-edge" in by_model and "bi-small" in by_model and all(
        by_model[name].get("process_f1") for name in ("bi-edge", "bi-small")
    ):
        edge, small = by_model["bi-edge"], by_model["bi-small"]
        gain = float(small["process_f1"]) - float(edge["process_f1"])
        latency = _relative(float(small["median_latency_ms"]), float(edge["median_latency_ms"]))
        ram = _relative(float(small["model_ram_mb"]), float(edge["model_ram_mb"]))
        lines += [
            "",
            "## Trade-off",
            "",
            f"`bi-small` changes exact process F1 by {gain:+.3f}, median latency by {latency:+.1f}%, "
            f"and incremental model RAM by {ram:+.1f}% relative to `bi-edge`.",
            "",
            "Apply the deployment limits in `configs/models.yaml` before making the final go/no-go decision. "
            "When both models satisfy the limits, the benchmark policy selects `bi-edge` unless the measured "
            "quality gain from `bi-small` is materially important for downstream use.",
        ]
    lines += _decision_lines(rows, decision_criteria or {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(summary_path: Path, runs_path: Path, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open(encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    plottable = [row for row in summary if row.get("process_f1") and row.get("median_latency_ms")]
    if plottable:
        fig, axis = plt.subplots(figsize=(6.5, 4))
        for row in plottable:
            axis.scatter(float(row["median_latency_ms"]), float(row["process_f1"]), s=70, label=row["model"])
        axis.set(xlabel="Median latency (ms)", ylabel="Exact process micro F1", title="Accuracy–latency trade-off")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_latency.png", dpi=160)
        plt.close(fig)
    with runs_path.open(encoding="utf-8") as handle:
        runs = [row for row in csv.DictReader(handle) if row["split"] == "dev" and row["slice"] == "overall"]
    if runs:
        fig, axis = plt.subplots(figsize=(7, 4))
        for key in sorted({(row["model"], row["dataset"]) for row in runs}):
            values = sorted((float(row["threshold"]), float(row["exact_micro_f1"])) for row in runs if (row["model"], row["dataset"]) == key)
            axis.plot([x for x, _ in values], [y for _, y in values], marker="o", label=" / ".join(key))
        axis.set(xlabel="Threshold", ylabel="Exact micro F1", title="Development threshold sweep")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "threshold_sweep.png", dpi=160)
        plt.close(fig)


def _relative(new: float, old: float) -> float:
    return (new / old - 1) * 100 if old else 0.0


def _decision_lines(rows: list[dict], criteria: dict) -> list[str]:
    required = {
        "minimum_process_f1": ("process_f1", lambda actual, limit: actual >= limit),
        "minimum_pii_recall": ("pii_recall", lambda actual, limit: actual >= limit),
        "maximum_p95_latency_ms": ("p95_latency_ms", lambda actual, limit: actual <= limit),
        "maximum_model_ram_mb": ("model_ram_mb", lambda actual, limit: actual <= limit),
    }
    if not criteria or any(criteria.get(name) is None for name in required):
        return [
            "",
            "## Deployment decision",
            "",
            "Not automated: one or more deployment limits in `configs/models.yaml` are unset.",
        ]
    lines = ["", "## Deployment decision", ""]
    passing = []
    for row in rows:
        checks = []
        for criterion, (column, comparison) in required.items():
            value = row.get(column)
            checks.append(bool(value) and comparison(float(value), float(criteria[criterion])))
        if all(checks):
            passing.append(row["model"])
        lines.append(f"- `{row['model']}`: {'passes' if all(checks) else 'fails'} the configured limits")
    if "bi-edge" in passing:
        lines += ["", "Recommendation: deploy `bi-edge` unless the documented `bi-small` quality gain is materially necessary."]
    elif "bi-small" in passing:
        lines += ["", "Recommendation: `bi-edge` misses the configured limits; `bi-small` is the remaining viable candidate."]
    else:
        lines += ["", "Recommendation: neither model satisfies the configured deployment limits."]
    return lines
