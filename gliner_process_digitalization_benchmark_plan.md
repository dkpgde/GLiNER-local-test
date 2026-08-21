# GLiNER Benchmark Plan — Process Improvement & Digitalization Interviews

## 1. Objective

Benchmark whether a compact GLiNER model is useful enough to justify running locally on a resource-constrained server as part of process-improvement and digitalization interviews.

The benchmark should answer:

1. How well can GLiNER identify entities useful to process analysis and digitalization?
2. How well can the same model detect PII for optional redaction?
3. What CPU latency and RAM cost does that capability impose under a strict single-thread constraint?
4. Is the second-smallest model materially better enough to justify its additional resource usage?

The likely deployment candidate is the smallest model. The second-smallest model is included mainly to quantify the accuracy–resource trade-off.

---

## 2. Models

Benchmark:

- `knowledgator/gliner-bi-edge-v2.0`
- `knowledgator/gliner-bi-small-v2.0`

Use:

- identical preprocessing
- identical label definitions
- identical threshold sweeps
- identical datasets
- identical single-thread CPU conditions

No fine-tuning in the initial benchmark.

Where supported, precompute and cache label embeddings because the production label set will be fixed.

---

## 3. Deployment Constraint

All benchmark inference must use one CPU thread.

Configure explicitly, for example:

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

and, where applicable:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Do not benchmark 2-core, 4-core, or unrestricted-thread configurations.

The purpose is to approximate deployment on a server where GLiNER must coexist with other services and cannot assume spare CPU capacity.

---

## 4. Evaluation Tracks

### Track A — Process/Digitalization NER

Primary domain benchmark.

Use ProcessChat as source material and create a manually annotated evaluation subset.

Target:

- 150–300 representative dialogue turns

Sample deliberately for:

- actors and roles
- departments and organizations
- activities
- handoffs
- business objects
- documents
- software and applications
- data fields and identifiers
- timing and frequency
- manual work
- exceptions
- pain points
- bottlenecks
- multiple relevant entities in one utterance

### Track B — Conversational / PII NER

Use DialogPII.

Purpose:

- test entity extraction in conversational text
- measure PII detection quality
- evaluate whether GLiNER could also serve as a privacy-redaction layer
- compare performance on cleaner and noisier transcript conditions where available

Keep the original DialogPII labels and additionally report a privacy-focused subset.

---

## 5. Process/Digitalization Entity Schema

### Stakeholders

- `person`
- `role`
- `department`
- `organization`
- `customer`
- `supplier`

### Process Content

- `activity`
- `process`
- `event`
- `business_object`
- `document`

### Technology

- `software_system`
- `application`
- `machine`
- `device`

### Data

- `data_field`
- `identifier`
- `database`
- `file_format`

### Operational Parameters

- `location`
- `time_duration`
- `frequency`
- `quantity`
- `money`

### Improvement Signals

- `manual_step`
- `pain_point`
- `bottleneck`
- `exception`

### Privacy

- `email`
- `phone`
- `address`
- `personal_identifier`
- other relevant PII labels supported by DialogPII

---

## 6. Scope Boundary

The benchmark evaluates span extraction only.

Do not treat GLiNER as responsible for:

- process sequence reconstruction
- control flow
- gateways as process structure
- actor → activity relations
- system → activity relations
- input/output relations
- causal reasoning
- BPMN generation
- recommendation generation

Those belong to downstream reasoning components.

---

## 7. ProcessChat Gold Set

Create a small manually annotated gold-standard subset.

Internal format:

```json
{
  "id": "processchat_001",
  "text": "The purchasing clerk enters the order into SAP.",
  "entities": [
    {
      "start": 4,
      "end": 20,
      "text": "purchasing clerk",
      "label": "role"
    },
    {
      "start": 32,
      "end": 37,
      "text": "order",
      "label": "business_object"
    },
    {
      "start": 43,
      "end": 46,
      "text": "SAP",
      "label": "software_system"
    }
  ]
}
```

Before full annotation, write a short annotation guide covering ambiguous cases such as:

- `activity` vs `process`
- `business_object` vs `document`
- `role` vs `department`
- `software_system` vs `application`
- `manual_step` vs ordinary `activity`
- `pain_point` vs `bottleneck`

Prefer consistency over exhaustive labeling.

---

## 8. Accuracy Metrics

### ProcessChat

Report:

- exact-span precision
- exact-span recall
- exact-span micro F1
- macro F1
- per-label F1
- relaxed/overlap F1

Also aggregate by functional group:

- stakeholder F1
- process-content F1
- technology F1
- data F1
- operational-parameter F1
- improvement-signal F1

### DialogPII

Report two views.

#### Full conversational NER

- micro F1
- macro F1
- per-class precision / recall / F1
- exact-span F1
- relaxed/overlap F1

#### Privacy subset

- PII precision
- PII recall
- PII F1
- false negatives per 1,000 PII entities

For redaction suitability, prioritize recall over precision.

---

## 9. Threshold Evaluation

Run a small threshold sweep for both models, for example:

- 0.2
- 0.3
- 0.4
- 0.5
- 0.6

Evaluate whether one threshold works acceptably across both tasks.

If not, report separate operating points for:

- process/digitalization NER
- privacy-oriented PII detection

Do not tune on the final evaluation split.

---

## 10. Performance Measurement

Measure efficiency during the same benchmark runs used for accuracy evaluation.

For every inference record:

- input token count
- inference latency
- model
- threshold
- dataset
- example ID

From those records derive:

- median latency
- p95 latency
- tokens/second
- documents/second
- latency by input-length bucket

Also record:

- model load time
- model size on disk
- baseline process RSS before model load
- RSS after model load
- peak RSS during inference

No separate synthetic performance benchmark is required initially.

The real DialogPII and ProcessChat input distributions are the relevant workload.

---

## 11. Main Comparison

Final summary table:

| Model | Process NER F1 | Technology F1 | Improvement Signal F1 | PII Recall | PII F1 | Median Latency | p95 Latency | Model RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bi-edge | | | | | | | | |
| bi-small | | | | | | | | |

Also report:

- absolute F1 gain of `bi-small`
- relative latency increase
- relative RAM increase
- whether `bi-small` fits within the realistic server resource budget

The comparison is intentionally asymmetric:

- `bi-edge` is the main deployment candidate
- `bi-small` is the quality ceiling for the smallest practical model class

If `bi-small` does not fit comfortably, that is still a useful benchmark result.

---

## 12. Error Analysis

Inspect at least:

- 50 false negatives
- 50 false positives

Classify recurring failures.

Useful categories:

### False negatives

- ambiguous span
- unusual wording
- transcript noise
- rare identifier
- implicit activity
- compound business term
- boundary error
- label confusion

### False positives

- ordinary business term
- role mistaken for person
- organization mistaken for system
- activity mistaken for pain point
- document/business-object confusion
- generic process language

Error analysis should focus on whether failures would materially harm downstream process understanding.

---

## 13. Repository Structure

```text
gliner-interview-benchmark/
├── data/
│   ├── raw/
│   │   ├── dialogpii/
│   │   └── processchat/
│   └── processed/
│       ├── dialogpii.jsonl
│       └── processchat_gold.jsonl
├── src/
│   ├── loaders/
│   │   ├── dialogpii.py
│   │   └── processchat.py
│   ├── inference.py
│   ├── metrics.py
│   ├── profiling.py
│   └── benchmark.py
├── configs/
│   ├── labels.yaml
│   └── models.yaml
├── results/
│   ├── predictions/
│   ├── metrics/
│   └── summary.csv
├── notebooks/
│   └── error_analysis.ipynb
├── requirements.txt
└── README.md
```

Keep the implementation small and benchmark-oriented.

---

## 14. Implementation Order

1. Create repository structure.
2. Implement single-thread GLiNER wrapper.
3. Implement fixed/cached label embeddings where supported.
4. Implement common span format.
5. Load DialogPII.
6. Implement metrics.
7. Run first DialogPII benchmark for both models.
8. Load ProcessChat.
9. Write annotation guide.
10. Annotate 150–300 ProcessChat turns.
11. Run ProcessChat benchmark.
12. Integrate latency and RSS recording into inference.
13. Run threshold sweeps.
14. Perform error analysis.
15. Produce final comparison table and plots.
16. Decide whether `bi-edge` is viable and whether `bi-small` offers enough quality to justify its additional resource cost.

---

## 15. Decision Criteria

The benchmark should ultimately support one deployment decision:

> Is `gliner-bi-edge-v2.0` useful enough at one CPU thread and acceptable RAM usage to justify running as a local interview-processing component?

Secondary question:

> If not, or if its accuracy is marginal, does `gliner-bi-small-v2.0` improve process-domain NER enough to justify its higher memory and latency cost?

The smallest model should win by default unless the larger model produces a clearly material improvement in process-relevant extraction quality.
