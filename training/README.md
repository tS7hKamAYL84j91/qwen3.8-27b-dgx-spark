# Training Program

This directory holds the post-training program for the resident Qwen3.8-27B
runtime: fine-tuning and distillation specs that would specialize the model
for administrative and Quartermaster-style (structured, tool-driven) tasks.

## Documents

| Spec | Title | Status |
| --- | --- | --- |
| [0001](specs/0001-qlora-sft-finetuning.md) | QLoRA SFT fine-tuning on DGX Spark | Draft — pending feasibility gate |
| [0002](specs/0002-distillation-pipeline.md) | Teacher distillation, on-policy self-distillation, drafter re-alignment | Draft |

## Program shape

The two specs are sequential, not alternatives:

1. **Spec 0001** defines the training substrate: QLoRA SFT with Unsloth on
   this DGX Spark, the eval harness, and the shared dataset/validator schema.
2. **Spec 0002** defines where the training signal comes from: black-box
   distillation from cloud teachers (primary), on-policy self-distillation
   with execution feedback (conditional Stage 2), and re-aligning the DSpark
   drafter to the distilled target so the speculative speedup survives
   (Stage 3).

## Hard gates

- **Feasibility gate:** a 30-minute smoke test must confirm the hybrid
  `qwen3_5` (Mamba + attention) architecture trains through the chosen stack
  before any full run (spec 0001, §7).
- **Governance gate:** a behaviorally modified model serving coas agents is a
  residency-class decision. Executing this program requires Gravitas sign-off,
  recorded as ADR-0003. Training may not begin on spec drafts alone.
- **Residency gate:** training windows require the GPU exclusively
  (`make stop` → train → `make setup`), per the memory analysis in ADR-0001;
  SGLang serving and training cannot coexist on 121.7 GiB unified memory.

## Non-goals

- Full-parameter fine-tuning (infeasible on this box; unnecessary for task
  specialization).
- Distilling into a smaller student model (changes the capability envelope;
  separate decision if ever raised).
- Redistribution of teacher-generated data (internal use only; cloud-teacher
  terms do not cover redistribution).
