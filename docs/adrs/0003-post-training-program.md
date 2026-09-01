# ADR-0003: Execute the admin/Q post-training program (minimal path)

- Status: Proposed
- Date: 2026-09-01
- Owner: qwen3.8-27b-dgx-spark GM
- Related: ADR-0001 (residency), ADR-0002 (serving config),
  training/specs/0001 (QLoRA SFT), training/specs/0002 (distillation)

## Context

The Executive Office wants the resident Qwen3.8-27B to reliably perform
administrative and Quartermaster-style tasks: schema-valid tool calls,
structured reports, and workflow execution. The base model has no labeled
admin/Q training data, and its general-purpose training does not guarantee
the ≥98% tool-call fidelity and template fidelity these tasks need.

The proposed program (training/specs/0001 and 0002) is a single minimal
pipeline, not two projects:

1. Teacher distillation: cloud teachers (glm-5.3, kimi-k2.7-code,
   deepseek-v4-flash per the coas routing policy) generate admin/Q task
   traces; deterministic validators filter them into a 10-20k corpus.
2. One QLoRA SFT run (Unsloth on DGX Spark, ~30-40 GiB, hours on-box).
3. Eval against the capability targets in spec 0001 §2.

On-policy self-distillation (spec 0002 Stage 2) stays conditional on Stage 1
plateauing. DSpark drafter re-alignment (spec 0002 Stage 3) is mandatory only
if the fine-tuned model is promoted to serving, to preserve the ~3.39 mean
acceptance length and current decode throughput.

## Decision

Execute the minimal path:

- Run the spec-0001 §7 feasibility smoke test (hybrid `qwen3_5` architecture
  through the Unsloth/TRL stack) before any data work is committed.
- Run Stage 1 distillation + one QLoRA SFT run + eval.
- Defer Stage 2 (on-policy self-distillation) and the Stage 3 drafter
  re-alignment decision to eval evidence; Stage 3 becomes mandatory only at
  promotion time.

## Alternatives considered

- **Prompt-level only** (no training): viable only if the admin/Q task
  surface is narrow; rarely sustains the 98% tool-call fidelity target on
  complex tool schemas. Cheap probe; revisit if the task inventory is small.
- **Full program (add OPD and drafter re-alignment up front):** rejected —
  adds complexity and training windows before evidence justifies them.
- **No action:** accepted risk is that the local model remains the weakest
  link in admin/Q automation; tasks fall back to cloud models (quota burn)
  or human effort.

## Consequences

- Training windows take the GPU exclusively; serving downtime per window is
  the same cost already accepted for benchmarking (ADR-0001/0002).
- A promoted fine-tuned model changes coas agent behavior; the serving
  switch itself (checkpoint pinning + drafter swap) needs its own ADR entry
  referencing this one.
- Teacher-generated corpus is internal-use only (cloud-teacher terms do not
  cover redistribution).
- If the feasibility smoke test fails on the hybrid architecture, the program
  falls back to LLaMA-Factory for the training substrate before any data
  spend.

## Status

Proposed, pending Gravitas approval. If approved, execution order is:
smoke test → admin/Q task inventory from coas → Stage 1 data build →
single SFT run → eval → promotion decision (which triggers Stage 3).
