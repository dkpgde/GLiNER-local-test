# ProcessChat annotation guide

Annotate useful, explicit spans only. Character offsets use Python slicing:
`text[start:end]` must equal the stored entity text. Overlapping spans are
allowed only when both annotations add clear downstream value. Consistency is
more important than capturing every possible noun phrase.

## Boundary rules

- Include the complete identifying noun phrase but omit determiners: annotate
  `purchasing clerk`, not `the purchasing clerk`.
- Include modifiers that change the business meaning (`purchase order`,
  `customer account number`), but not incidental adjectives.
- Annotate repeated mentions independently.
- Do not infer omitted actors, systems, activities, causes, or relations.

## Ambiguous labels

### `activity` vs `process`

Use `activity` for one task or action (`checks the invoice`). Use `process` for
a named or clearly end-to-end workflow (`employee onboarding process`). A verb
phrase is not automatically a process.

### `business_object` vs `document`

Use `document` for a human-readable or formally issued artifact such as an
invoice, form, report, or contract. Use `business_object` for an operational
item or record such as an order, claim, ticket, shipment, or case. If the text
clearly refers to a file or form, prefer `document`.

### `role` vs `department`

Use `role` for a job, responsibility, or actor type (`claims adjuster`). Use
`department` for an organizational unit (`Accounts Payable`). Named companies
and institutions are `organization`.

### `software_system` vs `application`

Use `software_system` for a platform or enterprise system (`SAP`, `the ERP
system`). Use `application` for a user-facing app or discrete software tool
(`mobile banking app`, `Excel`). Apply the distinction consistently even when
the boundary is imperfect.

### `manual_step` vs ordinary `activity`

Use `manual_step` only when the wording makes human/manual execution salient
(`copy it by hand`, `manually re-enter`). Do not label every human activity as
manual. Label the same span as `activity` as well only when nested/overlapping
annotations are being used consistently throughout the set.

### `pain_point` vs `bottleneck`

Use `pain_point` for an expressed problem, frustration, error-prone condition,
or unnecessary effort. Use `bottleneck` when capacity, waiting, queuing, or a
specific constraint limits flow. A bottleneck can also be a pain point, but
prefer the more specific label unless double annotation is a deliberate rule.

### `exception`

Use for an explicit departure from the normal path (`if the amount is over
€10,000`, `when the record is missing`). Do not annotate the entire surrounding
sentence when a shorter condition is sufficient.

## Quality check

Before freezing the gold set:

1. Validate every offset with `python -m src.schema data/processed/processchat_gold.jsonl`.
2. Review at least 20 shared examples with a second annotator or in a calibration
   pass.
3. Freeze the development/test split already present in the template.
4. Never change labels or thresholds after inspecting held-out test results.
