# Spec 0002: Distillation pipeline for admin/Q tasks

- Status: Draft (execution gated by ADR-0003)
- Depends on: spec 0001 (training substrate, dataset schema, validators)
- Evidence anchors: DSpark drafter card (mean acceptance length 3.39 over
  1,164 requests; target aux layers 4/16/28/40/52); SpecForge DSpark offline
  colocated configs; OPD failure conditions from arXiv 2604.13016 (2026).

## Stage 1 — Black-box task distillation (primary)

Sequence-level distillation: strong teachers generate admin/Q task traces;
the distilled traces become the SFT corpus for spec 0001's QLoRA run.

1. **Teacher pool:** the cloud teachers already routed via ollama
   (`glm-5.3:cloud`, `kimi-k2.7-code:cloud`, `deepseek-v4-flash:cloud`),
   chosen per the coas routing policy (no Pro-tier burn). Black-box only:
   these APIs expose text, not logits, so logit-level KD is out of scope.
2. **Generation:** temperature 0.7, 2-4 samples per prompt, max 4-8k tokens.
   Admin/Q traces are short-horizon, which is where distillation signal is
   strongest (OPD literature: reward reliability degrades on long horizons).
3. **Filtering:** deterministic validators from the spec-0001 schema; reject
   invalid tool calls and template violations; dedup; keep top band.
4. **Corpus target:** 10-20k validated traces covering both task families,
   mixed difficulty, 95/5 train/eval split.

Rationale: strongest teachers + no labeled data = classic distillation case.
R1-Distill-style sequence distillation delivered 30-60 point gains on
reasoning benchmarks at every model size; the mechanism transfers to
tool-calling/structured-output behavior.

## Stage 2: on-policy self-distillation (conditional)

Only if Stage 1 plateaus below the eval threshold.

- **Pattern:** on-policy self-distillation with privileged information
  (SDPO/OPSD family): the model is its own teacher conditioned on execution
  feedback (tool results, validator outcomes). Cross-family cloud teachers
  cannot provide token-level signals, and the OPD literature shows
  cross-family teachers can fail outright despite higher benchmark scores.
- **Conditions honored:** off-policy cold start is Stage 1; self-teacher
  guarantees thinking-pattern consistency; admin/Q traces are short-horizon,
  inside the length band where token-level reward stays reliable.
- **Cost control:** prefix-OPD (supervise only the first N tokens of each
  rollout) preserves quality at 2-40x lower FLOP; EMA self-teacher per SDPO.
- **Gate:** run only if Stage 1 eval plateaus; token-level signals require
  logprob access via our SGLang serving path.

## Stage 3: DSpark drafter re-alignment (mandatory before promotion)

The DSpark drafter is a distilled artifact (1.36B, SpecForge-trained against
target hidden layers 4/16/28/40/52, confidence head, block 7). A fine-tuned
target shifts the token distribution, so the drafter's acceptance length
(baseline: 3.39 mean across the drafter card's 11 workloads) will decay,
pushing decode toward the ~18 tok/s no-speculation floor.

- **Tooling:** SpecForge supports DSpark training (offline colocated and
  online disaggregated recipes exist). Warm-start from the RadixArk
  checkpoint; train draft-only with precomputed target hidden states.
- **Data:** regenerate the admin/Q corpus through the *fine-tuned* target
  (SpecBundle evidence: regenerating responses through the target improves
  acceptance ~5% over raw corpora).
- **Export:** `specforge export --to hf` (the DSpark serving-key contract),
  then serve and re-measure acceptance with the existing bench methodology.
- **Promotion bar:** mean acceptance length within 10% of the 3.39 baseline
  on the admin/Q workload mix, or an explicit acceptance of a lower bar
  recorded in the ADR.

## Compute windows and residency

| Stage | Est. duration | Memory | Notes |
| --- | --- | --- | --- |
| Teacher generation | 1-2 days (API latency) | negligible | cloud quota per routing policy |
| SFT (QLoRA) | hours per epoch | ~30-40 GiB | `make stop` window |
| OPD (if run) | ~4 h per 1k steps (20B reference; 27B somewhat more) | ~40 GiB | `make stop` window |
| Drafter retrain | hours (1.36B) | draft-only + cached hidden states | offline mode; hidden-state precompute needs target forwards |

## Open questions

1. Exact admin/Q task inventory and tool surface (from the coas side) —
   required to finalize the prompt set and validators.
2. Cloud-teacher generation quota: batch within the routing policy's quota
   guidance, or generate with a temporarily-resident local teacher instead.
3. Whether Stage 2 self-distillation is worth the added complexity — decide
   on Stage 1 eval results, not by default.
