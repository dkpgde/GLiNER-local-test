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

## 4. Error analysis

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
server budget. The smallest model wins by default. Select `bi-small` only when
it fits comfortably and its measured process-domain gain is material for the
downstream workflow. PII redaction should be judged primarily by held-out recall
and false negatives per 1,000 direct identifiers.

## Data and result handling

Raw data, processed JSONL, predictions, and metrics are gitignored. This avoids
redistributing licensed datasets and accidentally publishing interview or PII
content. Review `data/README.md` before sharing any benchmark artifact.
