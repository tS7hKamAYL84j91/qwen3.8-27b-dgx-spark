# Spec 0001: QLoRA SFT Fine-Tuning on DGX Spark

- Status: Draft — pending feasibility gate (§7) and Gravitas sign-off (ADR-0003)
- Date: 2026-09-01
- Owner: qwen3.8-27b-dgx-spark GM
- Related: ADR-0001 (residency), ADR-0002 (serving config), spec 0002

## 1. Objective

Specialize the resident Qwen3.8-27B for administrative and Quartermaster-style
tasks: reliable tool-call emission, schema-valid structured outputs, and
faithful execution of office workflow formats (status reports, checklists,
state/log updates), without degrading general capability or serving speed.

## 2. Capability targets (definition of "meets admin and Q tasks")

| Capability | Metric | Target |
| --- | --- | --- |
| Tool-call fidelity | % of emitted tool calls that parse and validate against the office tool schemas | ≥ 98% |
| Task completion | % of held-out admin/Q tasks completed end-to-end (validator-verified) | ≥ 90% of a validated cloud-teacher baseline on the same suite |
| Structured output fidelity | % of reports/logs matching the required template byte-schema | ≥ 98% |
| General capability retention | MT-Bench-style spot set, ≤ 2% regression vs base model | pass |
| Serving perf | bench.py decode tok/s within 5% of the promoted profile; prefill unchanged | pass |

## 3. Base checkpoint

Fine-tune from `Qwen/Qwen3.8-27B-FP8` (or the BF16 checkpoint). The locally
present `radix-nvfp4` is a quantization artifact and is not a training base.
After training, merge LoRA and export; serving quantization is decided at
deployment (FP8 direct, or NVFP4 requantization — see spec 0002, Stage 3
dependency).

## 4. Method: QLoRA SFT

- **Framework:** Unsloth + TRL on DGX Spark (NVIDIA official playbook:
  `nvcr.io/nvidia/pytorch:25.11-py3`, GB10-built Triton and xformers, TRL,
  PEFT, bitsandbytes). This is the NVIDIA-documented path for this hardware.
- **Adapters:** QLoRA 4-bit (NF4) load, LoRA r=16-32, alpha=2r, dropout 0.05,
  targets: attention projections + MLP projections (no embedding/LM-head).
- **Hyperparameters (starting point):** lr 1-2e-4 cosine, warmup 5%, 2-3
  epochs, effective batch 16-32 via gradient accumulation, bf16, seq len 4096
  (8192 for long report traces), gradient checkpointing on.
- **Memory budget:** ~30-40 GiB unified memory for a 27B QLoRA run
  (gpt-oss-120B QLoRA is documented at ~68 GiB on this class of hardware),
  comfortably inside 121.7 GiB with SGLang stopped.

## 5. Dataset and validator schema (shared with spec 0002)

One JSONL record per example:

```json
{
  "task_id": "adm-0042",
  "domain": "admin | quartermaster",
  "task_type": "status-report | schedule | inventory-update | agent-handoff | doc-format",
  "prompt": "...",
  "context": {"tools": [...], "prior_state": {}},
  "reference_trajectory": [
    {"role": "assistant", "tool_call": {"name": "...", "args": {}}},
    {"role": "tool", "result": "..."}
  ],
  "final_output": "schema-valid report/markup string",
  "validator": {"kind": "json-schema | state-transition | template-match", "spec": {}},
  "difficulty": "S | M | L",
  "source": "coas-trace | teacher-generated | hand-written"
}
```

Validators are pure functions reused in three places: training-data filtering
(spec 0002 Stage 1), GRPO/OPD reward functions (spec 0002 Stage 2), and the
eval harness. Minimum set: tool-call schema validator, template fidelity
checker, state-transition checker for Quartermaster flows.

## 6. Eval harness

- **Task eval:** held-out golden set (5% of corpus, never trained on):
  tool-call validity, end-to-end task completion, template fidelity.
- **Retention check:** general-capability spot suite to detect catastrophic
  forgetting; threshold: no regression beyond 2% on the retained general mix.
- **Perf check:** `benchmarks/bench.py` decode tok/s must stay within noise of
  the ADR-0002 baseline; any drop is investigated before promotion.

## 7. Feasibility gate (blocker before any full run)

The architecture is hybrid (`model_type: qwen3_5`, Mamba + attention layers,
`Qwen3_5ForConditionalGeneration`). Verify in a single 30-minute smoke test:
load the FP8/BF16 checkpoint through the Unsloth/TRL stack, attach LoRA, run
50 training steps on a toy batch without NaNs or layer-type errors. If the
hybrid path fails, escalate: either a supported stack variant (transformers
version pinning) or fall back to the distillation-only data route with
full-parameter SFT of a LoRA via LLaMA-Factory (its 2026 releases track
hybrid support more closely).

## 8. Runbook shape

```text
training/
  Dockerfile              # nvcr.io/nvidia/pytorch:25.11-py3 + GB10 triton/xformers + unsloth/trl/peft
  scripts/
    smoke_test.py         # feasibility gate
    build_dataset.py      # schema validation, dedup, 95/5 split
    train_sft_qlora.py    # adapter training
    merge_and_export.py   # LoRA merge -> FP8/BF16 export
  eval/
    admin_q_suite.jsonl   # golden held-out tasks
    run_eval.py           # validator-based scoring + bench.py integration
```

Estimated runtime: dataset build 1-2 days (dominated by teacher generation
and validation), SFT a few hours per run on-box (QLoRA 27B), eval 1-2 hours.
