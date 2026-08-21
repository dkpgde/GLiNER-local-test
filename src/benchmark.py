from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.inference import GlinerRunner, configure_single_thread
from src.metrics import evaluate
from src.profiling import summarize
from src.reporting import write_comparison, write_plots
from src.schema import Example, Span, read_jsonl, write_jsonl


def _threshold_name(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_once(
    runner: GlinerRunner,
    examples: Sequence[Example],
    prompts: dict[str, str],
    threshold: float,
    dataset_name: str,
    model_alias: str,
) -> tuple[dict[str, list[Span]], list[dict], list[dict]]:
    predictions: dict[str, list[Span]] = {}
    profiles: list[dict] = []
    output_records: list[dict] = []
    for example in examples:
        result = runner.predict(example.text, prompts, threshold)
        predictions[example.id] = result.entities
        profile = {"id": example.id, **result.profile}
        profiles.append(profile)
        output_records.append(
            {
                "id": example.id,
                "dataset": dataset_name,
                "model": model_alias,
                "threshold": threshold,
                "text": example.text,
                "gold_entities": [span.to_dict() for span in example.entities],
                "predicted_entities": [span.to_dict() for span in result.entities],
                "split": example.split,
                "metadata": example.metadata,
                "profile": result.profile,
            }
        )
    return predictions, profiles, output_records


def _metric_slices(examples: Sequence[Example], predictions: dict[str, list[Span]], track: dict) -> dict:
    result = {
        "overall": evaluate(examples, predictions, track.get("groups"), track.get("privacy_labels")),
        "by_version": {},
    }
    versions = sorted({str(example.metadata.get("version")) for example in examples if example.metadata.get("version")})
    for version in versions:
        subset = [example for example in examples if example.metadata.get("version") == version]
        result["by_version"][version] = evaluate(subset, predictions, track.get("groups"), track.get("privacy_labels"))
    return result


def _row(dataset: str, model: str, split: str, threshold: float, slice_name: str, metrics: dict, profile: dict) -> dict:
    exact = metrics["exact"]["micro"]
    privacy = metrics.get("privacy", {}).get("exact", {}).get("micro", {})
    return {
        "dataset": dataset,
        "model": model,
        "split": split,
        "threshold": threshold,
        "slice": slice_name,
        "exact_micro_precision": exact["precision"],
        "exact_micro_recall": exact["recall"],
        "exact_micro_f1": exact["f1"],
        "exact_macro_f1": metrics["exact"]["macro_f1"],
        "relaxed_micro_f1": metrics["relaxed"]["micro"]["f1"],
        "pii_precision": privacy.get("precision", ""),
        "pii_recall": privacy.get("recall", ""),
        "pii_f1": privacy.get("f1", ""),
        "pii_false_negatives_per_1000": metrics.get("privacy", {}).get("false_negatives_per_1000", ""),
        "documents": profile["documents"],
        "tokens": profile["tokens"],
        "median_latency_ms": profile["median_latency_ms"],
        "p95_latency_ms": profile["p95_latency_ms"],
        "documents_per_second": profile["documents_per_second"],
        "tokens_per_second": profile["tokens_per_second"],
        "peak_rss_mb": profile["peak_rss_mb"],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(model: str, final: dict[str, dict], runner: GlinerRunner) -> dict:
    process = final.get("processchat")
    pii = final.get("dialogpii")
    process_metrics = process["metrics"]["overall"] if process else {}
    pii_metrics = pii["metrics"]["overall"] if pii else {}
    combined_profiles = [item for value in final.values() for item in value["profiles"]]
    efficiency = summarize(combined_profiles)
    return {
        "model": model,
        "process_f1": _format(process_metrics.get("exact", {}).get("micro", {}).get("f1")),
        "technology_f1": _format(process_metrics.get("groups", {}).get("technology", {}).get("exact", {}).get("micro", {}).get("f1")),
        "improvement_signal_f1": _format(process_metrics.get("groups", {}).get("improvement_signal", {}).get("exact", {}).get("micro", {}).get("f1")),
        "pii_recall": _format(pii_metrics.get("privacy", {}).get("exact", {}).get("micro", {}).get("recall")),
        "pii_f1": _format(pii_metrics.get("privacy", {}).get("exact", {}).get("micro", {}).get("f1")),
        "median_latency_ms": f"{efficiency['median_latency_ms']:.2f}",
        "p95_latency_ms": f"{efficiency['p95_latency_ms']:.2f}",
        "model_ram_mb": f"{runner.model_ram_mb:.1f}",
        "model_size_mb": "" if runner.model_size_mb is None else f"{runner.model_size_mb:.1f}",
        "model_load_seconds": f"{runner.load_time_seconds:.2f}",
        "baseline_rss_mb": f"{runner.baseline_rss_mb:.1f}",
        "loaded_rss_mb": f"{runner.loaded_rss_mb:.1f}",
        "peak_rss_mb": f"{efficiency['peak_rss_mb']:.1f}",
    }


def _format(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GLiNER models on one CPU thread.")
    parser.add_argument("--dialogpii", type=Path)
    parser.add_argument("--processchat", type=Path)
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--labels-config", type=Path, default=Path("configs/labels.yaml"))
    parser.add_argument("--model", action="append", help="Model alias from models.yaml; repeat to select multiple")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, help="Debug-only maximum examples per split")
    args = parser.parse_args()
    if not args.dialogpii and not args.processchat:
        parser.error("provide --dialogpii, --processchat, or both")

    configure_single_thread()
    model_config = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
    label_config = yaml.safe_load(args.labels_config.read_text(encoding="utf-8"))["tracks"]
    selected_models = args.model or list(model_config["models"])
    datasets = {
        name: read_jsonl(path)
        for name, path in (("dialogpii", args.dialogpii), ("processchat", args.processchat))
        if path
    }
    for name, examples in datasets.items():
        if not any(example.entities for example in examples):
            raise ValueError(f"{name} has no gold entities; finish annotation before benchmarking")
        missing = {"dev", "test"} - {example.split for example in examples}
        if missing:
            raise ValueError(f"{name} is missing required split(s): {', '.join(sorted(missing))}")

    run_rows: list[dict] = []
    summary_rows: list[dict] = []
    for model_alias in selected_models:
        if model_alias not in model_config["models"]:
            raise KeyError(f"unknown model alias {model_alias!r}")
        runner = GlinerRunner(model_config["models"][model_alias]["model_id"])
        final: dict[str, dict] = {}
        try:
            for dataset_name, all_examples in datasets.items():
                track = label_config[dataset_name]
                prompts = dict(track["prompts"])
                # Cache label embeddings and warm model kernels outside measured records.
                runner.predict(all_examples[0].text, prompts, 0.5)
                dev = [example for example in all_examples if example.split == "dev"]
                test = [example for example in all_examples if example.split == "test"]
                if args.limit:
                    dev, test = dev[: args.limit], test[: args.limit]
                schedules = [("dev", dev, float(value)) for value in model_config["thresholds"] if dev]
                schedules.append(("test", test, float(model_config["operating_points"][dataset_name])))
                for split, examples, threshold in schedules:
                    if not examples:
                        continue
                    predictions, profiles, output = _run_once(
                        runner, examples, prompts, threshold, dataset_name, model_alias
                    )
                    metrics = _metric_slices(examples, predictions, track)
                    performance = summarize(profiles)
                    stem = f"{dataset_name}_{model_alias}_{split}_t{_threshold_name(threshold)}"
                    write_jsonl(args.results_dir / "predictions" / f"{stem}.jsonl", output)
                    _write_json(
                        args.results_dir / "metrics" / f"{stem}.json",
                        {"dataset": dataset_name, "model": model_alias, "split": split, "threshold": threshold, "metrics": metrics, "performance": performance},
                    )
                    run_rows.append(_row(dataset_name, model_alias, split, threshold, "overall", metrics["overall"], performance))
                    for version, version_metrics in metrics["by_version"].items():
                        ids = {example.id for example in examples if example.metadata.get("version") == version}
                        version_profile = summarize(profile for profile in profiles if profile["id"] in ids)
                        run_rows.append(_row(dataset_name, model_alias, split, threshold, version, version_metrics, version_profile))
                    if split == "test":
                        final[dataset_name] = {"metrics": metrics, "profiles": profiles}
            summary_rows.append(_summary_row(model_alias, final, runner))
        finally:
            runner.close()

    runs_path = args.results_dir / "metrics" / "benchmark_runs.csv"
    summary_path = args.results_dir / "summary.csv"
    _write_csv(runs_path, run_rows)
    _write_csv(summary_path, summary_rows)
    write_comparison(
        summary_path,
        args.results_dir / "comparison.md",
        model_config.get("decision_criteria", {}),
    )
    write_plots(summary_path, runs_path, args.results_dir / "plots")
    print(f"benchmark complete; final comparison: {summary_path}")


if __name__ == "__main__":
    main()
