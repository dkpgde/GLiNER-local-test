from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import Sequence

from src.profiling import length_bucket, model_size_mb, process_peak_rss_mb, rss_mb
from src.schema import Span


_THREADS_CONFIGURED = False


def configure_single_thread() -> None:
    global _THREADS_CONFIGURED
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    if _THREADS_CONFIGURED:
        return
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    _THREADS_CONFIGURED = True


@dataclass
class InferenceResult:
    entities: list[Span]
    profile: dict


class GlinerRunner:
    def __init__(self, model_id: str):
        configure_single_thread()
        from gliner import GLiNER

        self.model_id = model_id
        self.baseline_rss_mb = rss_mb()
        started = time.perf_counter()
        self.model = GLiNER.from_pretrained(model_id, map_location="cpu")
        self.load_time_seconds = time.perf_counter() - started
        self.loaded_rss_mb = rss_mb()
        self.model_ram_mb = max(0.0, self.loaded_rss_mb - self.baseline_rss_mb)
        self.model_size_mb = model_size_mb(model_id)
        self._label_cache: dict[tuple[str, ...], object] = {}

    def _embeddings(self, prompts: Sequence[str]):
        key = tuple(prompts)
        if key not in self._label_cache:
            if not hasattr(self.model, "encode_labels"):
                raise RuntimeError(f"{self.model_id} does not support cached label embeddings")
            self._label_cache[key] = self.model.encode_labels(list(prompts), batch_size=8)
        return self._label_cache[key]

    def _token_count(self, text: str) -> int:
        tokenizer = self.model.data_processor.transformer_tokenizer
        return len(tokenizer.encode(text, add_special_tokens=True))

    def predict(self, text: str, canonical_to_prompt: dict[str, str], threshold: float) -> InferenceResult:
        prompts = list(canonical_to_prompt.values())
        embeddings = self._embeddings(prompts)
        prompt_to_canonical = {prompt.casefold(): canonical for canonical, prompt in canonical_to_prompt.items()}
        token_count = self._token_count(text)
        rss_before = rss_mb()
        started = time.perf_counter()
        raw = self.model.batch_predict_with_embeds(
            [text], embeddings, prompts, threshold=threshold, batch_size=1
        )[0]
        latency_ms = (time.perf_counter() - started) * 1000
        rss_after = rss_mb()
        entities = []
        for item in raw:
            label = prompt_to_canonical.get(str(item["label"]).casefold())
            if label is None:
                raise ValueError(f"model returned unknown label {item['label']!r}")
            entities.append(
                Span.from_dict(
                    {
                        "start": item["start"],
                        "end": item["end"],
                        "text": item.get("text", text[item["start"] : item["end"]]),
                        "label": label,
                        "score": item.get("score"),
                    },
                    text,
                )
            )
        return InferenceResult(
            entities,
            {
                "token_count": token_count,
                "latency_ms": latency_ms,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "process_peak_rss_mb": process_peak_rss_mb(),
                "length_bucket": length_bucket(token_count),
            },
        )

    def close(self) -> None:
        self._label_cache.clear()
        del self.model
        gc.collect()
