from __future__ import annotations

import math
import os
import statistics
from pathlib import Path
from typing import Iterable, Mapping

import psutil


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def process_peak_rss_mb() -> float:
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / (1024 * 1024)
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024
    return rss_mb()


def model_size_mb(model_id: str) -> float | None:
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(model_id, local_files_only=True))
        return sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file()) / (1024 * 1024)
    except Exception:
        return None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(records: Iterable[Mapping]) -> dict:
    rows = list(records)
    latencies = [float(row["latency_ms"]) for row in rows]
    total_seconds = sum(latencies) / 1000
    summary = {
        "documents": len(rows),
        "tokens": sum(int(row["token_count"]) for row in rows),
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "documents_per_second": len(rows) / total_seconds if total_seconds else 0.0,
        "tokens_per_second": sum(int(row["token_count"]) for row in rows) / total_seconds if total_seconds else 0.0,
        "peak_rss_mb": max((float(row["process_peak_rss_mb"]) for row in rows), default=0.0),
        "latency_by_length_bucket": {},
    }
    for bucket in ("<64", "64-127", "128-255", "256+"):
        values = [float(row["latency_ms"]) for row in rows if row["length_bucket"] == bucket]
        if values:
            summary["latency_by_length_bucket"][bucket] = {
                "documents": len(values),
                "median_latency_ms": statistics.median(values),
                "p95_latency_ms": _percentile(values, 0.95),
            }
    return summary


def length_bucket(token_count: int) -> str:
    if token_count < 64:
        return "<64"
    if token_count < 128:
        return "64-127"
    if token_count < 256:
        return "128-255"
    return "256+"
