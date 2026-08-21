# GLiNER local interview benchmark

A small, reproducible benchmark for deciding whether
`knowledgator/gliner-bi-edge-v2.0` is useful enough for local process-improvement
interview analysis on one CPU thread, and whether `gliner-bi-small-v2.0` earns
its extra memory and latency.

The benchmark covers exact and overlap span accuracy, functional process-entity
groups, direct-identifier PII recall, per-document latency, throughput, model
load time, on-disk size, current/peak RSS, clean-vs-transcript DialogPII slices,
threshold sweeps, and error-review exports. It deliberately does not evaluate
relations, process reconstruction, BPMN generation, or recommendations.

## Setup

Python 3.10 or newer is required. Use a CPU build of PyTorch appropriate for the
host, then install the remaining dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All inference paths set PyTorch inter-op and intra-op threads plus OMP, MKL,
OpenBLAS, and NumExpr thread limits to one. GPU execution and unrestricted CPU
execution are not exposed by the CLI.

## 1. Prepare DialogPII

Download `DialogPII.zip` from the [official Zenodo
record](https://doi.org/10.5281/zenodo.20863452), place it under
`data/raw/dialogpii`, and convert the official turn-local offsets:

```powershell
python -m src.loaders.dialogpii data/raw/dialogpii/DialogPII.zip `
  --output data/processed/dialogpii.jsonl `
  --language EN --version both
```

Original dialog and noisy speech-transcript turns remain distinguishable through
`metadata.version`. A deterministic split keeps both versions of the same dialog
on the same side of the development/test boundary.

## 2. Build and annotate the ProcessChat gold set

Download the [official IBM ProcessChat
repository](https://github.com/IBM/ProcessChat) only after reviewing its
research-use and GPL-3.0-derived licensing notice. Generate a balanced template
of assistant/user and short/medium/long turns:

```powershell
python -m src.loaders.processchat data/raw/processchat `
  --output data/processed/processchat_annotation.jsonl `
  --sample-size 250
```

Annotate `entities` using [the annotation guide](docs/annotation_guide.md), save
the frozen file as `data/processed/processchat_gold.jsonl`, and validate all
offsets:

```powershell
python -m src.schema data/processed/processchat_gold.jsonl
```

The loader extracts the latest user turn and expected assistant response from
ProcessChat's `{input, output}` records, removes exact duplicates, samples across
speaker/length strata, and freezes deterministic development/test assignments.
Manual annotation is intentionally not fabricated by this repository.

## 3. Run the benchmark

Edit the fixed held-out operating points in `configs/models.yaml` before looking
at test results. Then run both tracks and models:

```powershell
python -m src.benchmark `
  --dialogpii data/processed/dialogpii.jsonl `
  --processchat data/processed/processchat_gold.jsonl
```

For a quick pipeline check, add `--limit 5 --model bi-edge`. Do not use limited
runs for decisions. Label prompts, functional groups, and the configurable
direct-identifier privacy subset live in `configs/labels.yaml`.

The command precomputes label embeddings once per model/track, warms the model
outside measured records, runs every configured threshold on `dev`, and runs
only the fixed operating point on `test`. Outputs include:

- `results/predictions/*.jsonl`: gold spans, predictions, and per-record profile
- `results/metrics/*.json`: exact/relaxed, per-label/group, privacy, and slice metrics
- `results/metrics/benchmark_runs.csv`: every development/test run
- `results/summary.csv`: held-out model comparison
- `results/comparison.md`: resource/quality trade-off summary
- `results/plots/*.png`: threshold and accuracy/latency plots

## 4. Benchmark results

The completed held-out run used threshold `0.5` for ProcessChat and `0.4` for
DialogPII. ProcessChat test results cover 207 records and 999 gold spans;
DialogPII test results cover 4,669 turns and 7,896 gold spans. Latency is for
single-threaded CPU inference and is combined across both held-out tracks.

| Model | ProcessChat exact P/R/F1 | DialogPII exact F1 | Direct-PII P/R/F1 | Median / p95 latency | Model RAM | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `bi-edge` | 0.385 / 0.138 / 0.203 | **0.547** | **0.577** / 0.660 / **0.616** | **89 / 159 ms** | **432 MB** | **954 MB** |
| `bi-small` | **0.455** / **0.143** / **0.218** | 0.543 | 0.516 / **0.715** / 0.599 | 169 / 292 ms | 657 MB | 1,386 MB |

### Interpretation

- `bi-small` improves ProcessChat exact micro F1 by only 0.015 while increasing
  median latency by 89%, p95 latency by 84%, and incremental model RAM by 52%.
  The gain is too small to justify the larger model for the general workflow.
- ProcessChat recall is only 0.138-0.143. The models therefore miss most gold
  process entities at the fixed operating point. The largest group,
  `process_content`, reaches only 0.149 F1 with `bi-edge` and 0.143 with
  `bi-small`; improvement-signal F1 is 0.000 and 0.041 respectively.
- On direct identifiers, `bi-small` raises recall from 0.660 to 0.715, reducing
  false negatives from about 340 to 285 per 1,000 gold identifiers. That is
  still far too many misses for unattended PII redaction. Its added recall also
  lowers precision from 0.577 to 0.516 and F1 from 0.616 to 0.599.
- Speech-transcript noise causes a modest but consistent drop. Compared with
  original DialogPII text, exact F1 falls by 0.029 for `bi-edge` and 0.022 for
  `bi-small`; direct-PII recall falls by roughly 0.044 and 0.042.
- Several ProcessChat subgroup results are based on few test spans (33 for
  technology and 27 for improvement signals), so their model-to-model
  differences should be treated as directional rather than conclusive.

The current recommendation is `bi-edge` for local, human-reviewed exploratory
extraction. Neither model is a go for comprehensive process mining or automatic
PII redaction with the current prompts and thresholds. A formal automated
go/no-go decision remains unavailable because the deployment limits in
`configs/models.yaml` are unset.

The ProcessChat development sweep has its best exact micro F1 near threshold
`0.3` for both models; the fixed `0.5` test point is a precision-oriented choice
with much lower recall. Do not choose a new threshold from these held-out test
results. A follow-up should select the operating objective on development data
and use a fresh blind test set. Full values are in
[`results/summary.csv`](results/summary.csv),
[`results/metrics/benchmark_runs.csv`](results/metrics/benchmark_runs.csv), and
[`results/comparison.md`](results/comparison.md).

## 5. Error analysis

Choose a held-out prediction file and create a balanced review sheet:

```powershell
python -m src.error_analysis `
  results/predictions/processchat_bi-edge_test_t0p50.jsonl `
  --output results/error_analysis.csv --per-type 50
```

Use `notebooks/error_analysis.ipynb` or any spreadsheet editor to assign failure
categories and record whether each error materially affects downstream process
understanding.

## Decision rule

Populate the resource and accuracy limits in `configs/models.yaml` with the real
server budget. The smallest model wins by default. Select `bi-small` only if its
extra direct-identifier recall is materially valuable, its remaining miss rate
is acceptable for a human-reviewed workflow, and its additional resource use
fits comfortably. PII redaction should be judged primarily by held-out recall
and false negatives per 1,000 direct identifiers, not aggregate DialogPII F1.

## Data and result handling

Raw data, processed JSONL, predictions, and metrics are gitignored. This avoids
redistributing licensed datasets and accidentally publishing interview or PII
content. Review `data/README.md` before sharing any benchmark artifact.

## References

- Zaratiana, U., Tomeh, N., Holat, P., & Charnois, T. (2024). [GLiNER:
  Generalist Model for Named Entity Recognition using Bidirectional
  Transformer](https://doi.org/10.18653/v1/2024.naacl-long.300). *Proceedings
  of NAACL-HLT 2024*, 5364–5376.
- Stepanov, I., Shtopko, M., Vodianytskyi, D., & Lukashov, O. (2026). [The
  Million-Label NER: Breaking Scale Barriers with GLiNER
  bi-encoder](https://arxiv.org/abs/2602.18487). arXiv:2602.18487. Model cards:
  [`gliner-bi-edge-v2.0`](https://huggingface.co/knowledgator/gliner-bi-edge-v2.0)
  and
  [`gliner-bi-small-v2.0`](https://huggingface.co/knowledgator/gliner-bi-small-v2.0).
- Roller, R., Czehmann, V., Erman, D., et al. (2026). [DialogPII: A
  multilingual dataset of synthetic dialog transcripts to detect personal
  information](https://arxiv.org/abs/2606.30312). arXiv:2606.30312. [Dataset,
  version 1.0](https://doi.org/10.5281/zenodo.20863452).
- Gantayat, N., Saha, A., & Sindhgatta, R. (2025). [ProcessChat: A Dataset for
  Business Process Grounded Dialogs](https://doi.org/10.1145/3799830.3799868).
  *Proceedings of the 13th ACM IKDD International Conference on Data Science
  (CODS '25)*, 215–223. [Dataset repository](https://github.com/IBM/ProcessChat).
