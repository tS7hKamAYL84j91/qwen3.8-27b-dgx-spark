#!/usr/bin/env python3
"""QLoRA SFT for admin/Quartermaster specialization (spec 0001 sections 4 and 8).

Uses the same FastLanguageModel/SFTTrainer stack proven by smoke_test.py
(feasibility gate PASSED 2026-09-01: hybrid qwen3_5 trains through
Unsloth/TRL on GB10). Configuration follows spec 0001 section 4: 4-bit NF4
QLoRA, LoRA r=16-32 alpha=2r dropout 0.05 on attention + MLP projections,
lr 1-2e-4 cosine with 5% warmup, 2-3 epochs, effective batch 16-32 via
gradient accumulation, bf16, gradient checkpointing, seq len 4096 (8192 for
long report traces).

DEFAULTS TO --dry-run: validates hyperparameters and the built dataset
without importing torch/unsloth or touching the GPU. Real training runs
only with an explicit --train.

Run inside the qwen38-train container with -w OUTSIDE the repo (see the
Dockerfile gcc/spec-file note); the base checkpoint mounts at /model:

    python3 /workspace/scripts/train_sft_qlora.py                 # dry-run
    python3 /workspace/scripts/train_sft_qlora.py --train        # real run

Prerequisite: training/scripts/build_dataset.py has produced
sft_train.jsonl / sft_eval.jsonl. The training window takes the GPU
exclusively (ADR-0001 residency gate: stop SGLang first).

Exit codes: 0 ok; 1 config/dataset validation error (or failed training
run); 2 unusable paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Sibling module: resolves at runtime because sys.path[0] is this
# script's directory when invoked per the spec-0001 section-8 runbook
# (python3 .../scripts/<name>.py).
import spec_common as sc  # pyright: ignore[reportMissingImports]

TRAINING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = TRAINING_DIR / "dataset" / "sft_train.jsonl"
DEFAULT_EVAL_FILE = TRAINING_DIR / "dataset" / "sft_eval.jsonl"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "outputs" / "sft_qlora"
DEFAULT_MODEL_PATH = os.environ.get("SFT_MODEL_PATH", "/model")

# Spec 0001 section 4: LoRA on attention projections + MLP projections
# (no embedding/LM-head).
LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Spec 0001 section 4 bounds.
LORA_R_MIN, LORA_R_MAX = 16, 32
LR_MIN, LR_MAX = 1e-4, 2e-4
EFFECTIVE_BATCH_RANGE = (16, 32)
EPOCH_RANGE = (2, 3)
SEQ_LENGTHS = (4096, 8192)
# ADR-0003 Stage-1 corpus target: 10-20k records.
CORPUS_TARGET_MIN = 10000
CHARS_PER_TOKEN_EST = 4

# Below this many records the run is a pipeline check, not a capability run.
PIPELINE_CHECK_THRESHOLD = 512


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT (spec 0001 section 4). Dry-run by default; "
                    "pass --train for the real GPU run."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true",
                      help="run real training (requires GPU + qwen38-train container)")
    mode.add_argument("--dry-run", action="store_true",
                      help="validate config + dataset only (default)")
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL_PATH),
                        help=f"base checkpoint path ($SFT_MODEL_PATH or {DEFAULT_MODEL_PATH})")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE,
                        help=f"SFT train JSONL from build_dataset.py (default: {DEFAULT_TRAIN_FILE})")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE,
                        help=f"SFT eval JSONL from build_dataset.py (default: {DEFAULT_EVAL_FILE})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"output directory for checkpoints + summary (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--lora-r", type=int, default=16,
                        help=f"LoRA rank, spec range {LORA_R_MIN}-{LORA_R_MAX} (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default: 2 * r per spec)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help=f"lr, spec range {LR_MIN:.0e}-{LR_MAX:.0e} (default: 2e-4)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        help="lr scheduler (default: cosine)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="warmup ratio (default: 0.05)")
    parser.add_argument("--epochs", type=int, default=2,
                        help=f"epochs, spec range {EPOCH_RANGE[0]}-{EPOCH_RANGE[1]} (default: 2)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="override epochs with a fixed step count (default: -1 = use epochs)")
    parser.add_argument("--per-device-batch", type=int, default=1,
                        help="per-device train batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=16,
                        help=f"gradient accumulation steps; effective batch = per-device * this, "
                             f"spec range {EFFECTIVE_BATCH_RANGE[0]}-{EFFECTIVE_BATCH_RANGE[1]} (default: 16)")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help=f"max sequence length, spec choices {SEQ_LENGTHS} (default: 4096)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the train dataset size (0 = no cap; for GPU pipeline checks)")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset-num-proc", type=int, default=4,
                        help="TRL dataset preprocessing workers (default: 4)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="do not evaluate on the internal eval split during training")
    return parser.parse_args(argv)


def validate_config(args: argparse.Namespace) -> list[str]:
    """Errors per the spec-0001 section-4 hyperparameter bounds."""
    errors: list[str] = []
    if not LORA_R_MIN <= args.lora_r <= LORA_R_MAX:
        errors.append(f"--lora-r must be {LORA_R_MIN}-{LORA_R_MAX} (spec section 4), got {args.lora_r}")
    if not LR_MIN <= args.learning_rate <= LR_MAX:
        errors.append(
            f"--learning-rate must be within {LR_MIN:.0e}-{LR_MAX:.0e} (spec section 4), "
            f"got {args.learning_rate:.2e}"
        )
    if args.max_seq_length <= 0 or args.max_seq_length > 32768:
        errors.append(f"--max-seq-length must be in (0, 32768], got {args.max_seq_length}")
    if args.per_device_batch < 1 or args.grad_accum < 1:
        errors.append("--per-device-batch and --grad-accum must be >= 1")
    if args.epochs < 1 and args.max_steps <= 0:
        errors.append("--epochs must be >= 1 when --max-steps is unset")
    return errors


def config_warnings(args: argparse.Namespace, effective_batch: int) -> list[str]:
    warnings: list[str] = []
    if args.lora_alpha is not None and args.lora_alpha != 2 * args.lora_r:
        warnings.append(f"lora_alpha {args.lora_alpha} != 2*r ({2 * args.lora_r}); spec section 4 says alpha=2r")
    if args.max_seq_length not in SEQ_LENGTHS:
        warnings.append(f"--max-seq-length {args.max_seq_length} is not a spec choice {SEQ_LENGTHS} "
                        "(4096 standard; 8192 for long report traces)")
    if not EFFECTIVE_BATCH_RANGE[0] <= effective_batch <= EFFECTIVE_BATCH_RANGE[1]:
        warnings.append(f"effective batch {effective_batch} outside spec range "
                        f"{EFFECTIVE_BATCH_RANGE[0]}-{EFFECTIVE_BATCH_RANGE[1]}")
    if args.max_steps <= 0 and not EPOCH_RANGE[0] <= args.epochs <= EPOCH_RANGE[1]:
        warnings.append(f"--epochs {args.epochs} outside spec range {EPOCH_RANGE[0]}-{EPOCH_RANGE[1]}")
    if not 0 <= args.warmup_ratio < 0.5:
        warnings.append("--warmup-ratio looks unusual (spec section 4: 5%)")
    return warnings


def load_sft_dataset(path: Path, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Load and validate an SFT JSONL produced by build_dataset.py."""
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for line_number, record in sc.iter_jsonl(path):
            errors.extend(sc.validate_sft_record(record, f"{path.name}:{line_number}"))
            records.append(record)
    except (OSError, ValueError) as exc:
        return [], [f"cannot read {label} file {path}: {exc}"]
    return records, errors


def estimate_tokens(characters: int) -> int:
    return math.ceil(characters / CHARS_PER_TOKEN_EST)


def dry_run(args: argparse.Namespace) -> int:
    """Validate config + dataset. No torch/unsloth import, no GPU side effects."""
    errors = validate_config(args)
    if args.model is None or not args.model.exists():
        print(f"NOTE: model path {args.model} not present here (normal on the host; "
              "the container mounts it at /model)")
    if not args.train_file.exists():
        errors.append(f"train file not found: {args.train_file} "
                      "(run training/scripts/build_dataset.py first)")
    train_records, load_errors = load_sft_dataset(args.train_file, "train") if args.train_file.exists() else ([], [])
    errors.extend(load_errors)
    eval_records: list[dict[str, Any]] = []
    if args.eval_file.exists() and not args.skip_eval:
        eval_records, load_errors = load_sft_dataset(args.eval_file, "eval")
        errors.extend(load_errors)
    if errors:
        for error in errors:
            print(f"DRY-RUN FAIL: {error}", file=sys.stderr)
        return 1

    if args.limit > 0:
        train_records = train_records[: args.limit]
    if not train_records:
        print("DRY-RUN FAIL: no train records", file=sys.stderr)
        return 1

    train_ids = {r["task_id"] for r in train_records}
    eval_ids = {r["task_id"] for r in eval_records}
    overlap = train_ids & eval_ids
    if overlap:
        print(f"DRY-RUN FAIL: train/eval task_id overlap: {sorted(overlap)}", file=sys.stderr)
        return 1

    effective_batch = args.per_device_batch * args.grad_accum
    warnings = config_warnings(args, effective_batch)

    def size_of(record: dict[str, Any]) -> int:
        return sum(len(m["content"]) for m in record["messages"])

    train_sizes = [size_of(r) for r in train_records]
    eval_sizes = [size_of(r) for r in eval_records]
    all_sizes = sorted(train_sizes + eval_sizes)
    over_budget = sum(1 for size in all_sizes if estimate_tokens(size) > args.max_seq_length)

    steps_per_epoch = math.ceil(len(train_records) / effective_batch)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r

    print("QLoRA SFT dry-run (spec 0001 section 4): config + dataset validation")
    print("  mode: DRY-RUN (no GPU, no torch import); pass --train for the real run")
    print(f"  model: {args.model}")
    print(f"  lora: r={args.lora_r} alpha={lora_alpha} dropout={args.lora_dropout} "
          f"targets={','.join(LORA_TARGET_MODULES)} (attention + MLP projections)")
    print(f"  optim: lr={args.learning_rate:.2e} scheduler={args.scheduler} "
          f"warmup={args.warmup_ratio} epochs={args.epochs} max_steps={args.max_steps}")
    print(f"  batch: per_device={args.per_device_batch} grad_accum={args.grad_accum} "
          f"effective={effective_batch}")
    print(f"  seq: max_seq_length={args.max_seq_length} bf16=True grad_checkpointing=True")
    print(f"  dataset:")
    print(f"    train: {args.train_file} ({len(train_records)} records)")
    print(f"    eval:  {args.eval_file} ({len(eval_records)} records"
          + (", skipped" if args.skip_eval else "") + ")")
    print(f"    train/eval task_id overlap: {len(overlap)} (must be 0)")
    if all_sizes:
        print(f"    rendered size: median={all_sizes[len(all_sizes) // 2]} "
              f"max={all_sizes[-1]} chars (~{estimate_tokens(all_sizes[-1])} tokens at "
              f"{CHARS_PER_TOKEN_EST} chars/token)")
        print(f"    over max_seq_length estimate: {over_budget} of {len(all_sizes)} records "
              "(TRL truncates these; consider --max-seq-length 8192)")
    print(f"  estimated train steps: {total_steps} "
          f"({steps_per_epoch}/epoch x {args.epochs} epochs at effective batch {effective_batch})")
    if len(train_records) < PIPELINE_CHECK_THRESHOLD:
        print(f"  NOTE: {len(train_records)} train records is below the ADR-0003 Stage-1 "
              f"target ({CORPUS_TARGET_MIN//1000}k-20k); this is a pipeline check, not a "
              "capability run")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        probe = args.output_dir / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        print(f"  output: {args.output_dir} (writable)")
    except OSError as exc:
        # Not fatal for a dry-run: the real run happens inside the qwen38-train
        # container, where /workspace/outputs is writable (as root).
        print(f"  WARNING: output dir {args.output_dir} not writable from this "
              f"environment ({exc}); inside the qwen38-train container the default "
              "path is /workspace/outputs (writable there)")
    print("DRY-RUN PASS: config and dataset validate; ready for --train in the "
          "qwen38-train container (SGLang stopped per ADR-0001).")
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Real training run: same FastLanguageModel/SFTTrainer pattern as smoke_test.py."""
    errors = validate_config(args)
    if errors:
        for error in errors:
            print(f"TRAIN ABORT: {error}", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"TRAIN ABORT: base checkpoint not found at {args.model} "
              "(mount it at /model in the qwen38-train container)", file=sys.stderr)
        return 2
    train_records, load_errors = load_sft_dataset(args.train_file, "train")
    eval_records, load_errors_eval = load_sft_dataset(args.eval_file, "eval") if args.eval_file.exists() else ([], [])
    errors.extend(load_errors)
    errors.extend(load_errors_eval)
    if errors:
        for error in errors:
            print(f"TRAIN ABORT: {error}", file=sys.stderr)
        return 1
    if args.limit > 0:
        train_records = train_records[: args.limit]
    if not train_records:
        print("TRAIN ABORT: no train records", file=sys.stderr)
        return 1

    try:
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"TRAIN ABORT: missing training dependency (run inside the qwen38-train "
              f"container): {exc}")
        return 1

    effective_batch = args.per_device_batch * args.grad_accum
    for warning in config_warnings(args, effective_batch):
        print(f"WARNING: {warning}")
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r

    print(f"[sft] loading {args.model} (4-bit QLoRA NF4, max_seq_length={args.max_seq_length})",
          flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.model),
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(LORA_TARGET_MODULES),
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_records])
    eval_ds = (
        Dataset.from_list([{"messages": r["messages"]} for r in eval_records])
        if eval_records and not args.skip_eval
        else None
    )

    config_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_batch,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_seq_length,
        "packing": False,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "bf16": True,
        "report_to": [],
        "seed": args.seed,
        "dataset_num_proc": args.dataset_num_proc,
    }
    if args.max_steps > 0:
        config_kwargs["max_steps"] = args.max_steps
    else:
        config_kwargs["num_train_epochs"] = args.epochs
    if eval_ds is not None:
        config_kwargs["eval_strategy"] = "epoch"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**config_kwargs),
    )
    print(f"[sft] training: {len(train_records)} train records"
          + (f", {len(eval_records)} eval records" if eval_ds is not None else "")
          + f", effective batch {effective_batch}, lr {args.learning_rate:.2e} "
          f"{args.scheduler}", flush=True)
    trainer.train()

    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry]
    if any(loss != loss for loss in losses):  # NaN check (same as the smoke gate)
        print("TRAIN FAIL: NaN loss observed", file=sys.stderr)
        final_status = "failed"
    else:
        final_status = "completed"

    final_dir = args.output_dir / "adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "schema": "spec-0001-sft-summary-v1",
        "status": final_status,
        "model": str(args.model),
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file) if eval_ds is not None else None,
        "records": {"train": len(train_records), "eval": len(eval_records)},
        "config": {
            "lora_r": args.lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(LORA_TARGET_MODULES),
            "learning_rate": args.learning_rate,
            "scheduler": args.scheduler,
            "warmup_ratio": args.warmup_ratio,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "per_device_batch": args.per_device_batch,
            "grad_accum": args.grad_accum,
            "effective_batch": effective_batch,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
        },
        "loss": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "log_points": len(losses),
            "eval_first": eval_losses[0] if eval_losses else None,
            "eval_last": eval_losses[-1] if eval_losses else None,
        },
        "log_history": trainer.state.log_history,
        "adapter_dir": str(final_dir),
    }
    summary_path = final_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    if final_status != "completed":
        return 1
    if losses:
        print(f"SFT COMPLETE: steps logged={len(losses)} "
              f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"
              + (f", eval_loss {eval_losses[0]:.4f} -> {eval_losses[-1]:.4f}" if eval_losses else ""))
    print(f"  adapter: {final_dir}")
    print(f"  summary: {summary_path}")
    print("  next runbook step: merge_and_export.py (LoRA merge -> FP8/BF16 export), "
          "then eval/run_eval.py against the golden suite.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.train:
        return run_train(args)
    return dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
