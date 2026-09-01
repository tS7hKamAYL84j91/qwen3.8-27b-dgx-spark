#!/usr/bin/env python3
"""Feasibility gate for spec 0001 section 7 (ADR-0003 ratification condition 1).

Proves the hybrid qwen3_5 (Mamba + attention) architecture trains through the
Unsloth/TRL stack on GB10: load Qwen3.8-27B-FP8 in 4-bit QLoRA, attach LoRA
adapters, run 50 SFT steps on a toy admin-style batch. PASS requires 50 steps
completed with finite loss that ends below its first recorded value.

Run inside the qwen38-train container; the checkpoint mounts at /model.

Usage:
    python3 /workspace/scripts/smoke_test.py
"""

import os
import sys
import traceback

MODEL_PATH = os.environ.get("SMOKE_MODEL_PATH", "/model")
STEPS = 50
SEQ_LEN = 1024

TOY_SAMPLES = [
    {
        "instruction": (
            "Draft a one-line status update for the weekly office report "
            "covering runtime health and open decisions."
        ),
        "output": "STATUS runtime=healthy decisions=2 open owner=GM\n",
    },
    {
        "instruction": (
            "Log an inventory update: 3 crates of type-B moved from bay 2 to "
            "bay 5, chain of custody recorded."
        ),
        "output": '{"type": "inventory-update", "item": "type-B", "qty": 3, '
                  '"from": "bay-2", "to": "bay-5"}',
    },
] * 8


def main():
    try:
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"SMOKE FAIL: missing training dependency (run inside the "
              f"qwen38-train container): {exc}")
        return 1

    print(f"[smoke] loading {MODEL_PATH} (4-bit QLoRA on BF16 base; the FP8 "
          f"quantizer path failed gate 1 — see ADR history)...", flush=True)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=SEQ_LEN,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing=True,
        )
    except Exception as exc:
        print(f"SMOKE FAIL: model/adapter load error: {exc}")
        print(traceback.format_exc())
        return 1

    dataset = Dataset.from_dict({
        "prompt": [e["instruction"] for e in TOY_SAMPLES],
        "completion": [e["output"] for e in TOY_SAMPLES],
    })

    config = SFTConfig(
        output_dir="/workspace/outputs/smoke",
        max_steps=STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_length=SEQ_LEN,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=config,
        )
        trainer.train()
    except Exception as exc:
        print(f"SMOKE FAIL: training error: {exc}")
        print(traceback.format_exc())
        return 1

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    if not losses:
        print("SMOKE FAIL: no loss values logged")
        return 1
    if any(l != l for l in losses):  # NaN check
        print("SMOKE FAIL: NaN loss observed")
        return 1
    if losses[-1] >= losses[0]:
        print(f"SMOKE FAIL: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})")
        return 1
    print(f"SMOKE PASS: {STEPS} steps, loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({len(losses)} log points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())