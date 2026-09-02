#!/usr/bin/env python3
"""Build the SFT dataset from the spec-0001 section-5 corpus (spec section 8).

Pipeline per spec 0001:
  1. validate every corpus record against the section-5 schema (fail loudly;
     --drop-invalid quarantines instead),
  2. deduplicate by task_id and by exact record content,
  3. exclude any task_id that appears in the golden held-out suite
     (training/eval/admin_q_suite.jsonl) so the golden set is never trained
     on,
  4. assign quality tiers: gold = needs_validation is false OR >= 2 tool
     calls in the reference trajectory, silver = the rest,
  5. stratified 95/5 train/eval split over (domain, task_type, tier)
     strata, deterministic for a given --seed,
  6. render records to conversational "messages" for TRL SFTTrainer and
     write sft_train.jsonl / sft_eval.jsonl + manifest.json with tier
     counts.

Exit codes: 0 ok; 1 schema violation (or empty output), 2 unreadable input.

Usage:
  python3 training/scripts/build_dataset.py
  python3 training/scripts/build_dataset.py --corpus X.jsonl --out-dir DIR
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Sibling module: resolves at runtime because sys.path[0] is this
# script's directory when invoked per the spec-0001 section-8 runbook
# (python3 .../scripts/<name>.py).
import spec_common as sc  # pyright: ignore[reportMissingImports]

TRAINING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = TRAINING_DIR / "corpus" / "admin_q_corpus.jsonl"
DEFAULT_GOLDEN = TRAINING_DIR / "eval" / "admin_q_suite.jsonl"
DEFAULT_OUT_DIR = TRAINING_DIR / "dataset"

# Rendered records above this char count likely exceed the spec-0001 section-4
# 4096-token sequence budget at ~4 chars/token (use 8192 for those runs).
SEQ_WARN_CHARS = 4096 * 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, tier, split, and render the spec-0001 admin/Q corpus for SFT."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                        help=f"corpus JSONL (default: {DEFAULT_CORPUS})")
    parser.add_argument("--golden-eval", type=Path, default=DEFAULT_GOLDEN,
                        help=f"golden held-out suite, used for leakage exclusion "
                             f"(default: {DEFAULT_GOLDEN})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--eval-fraction", type=float, default=0.05,
                        help="internal train/eval split fraction (default: 0.05)")
    parser.add_argument("--seed", type=int, default=0,
                        help="split seed; deterministic given corpus + seed (default: 0)")
    parser.add_argument("--drop-invalid", action="store_true",
                        help="quarantine schema-invalid records instead of failing")
    parser.add_argument("--max-tool-result-chars", type=int, default=0,
                        help="clamp each rendered tool result to N chars (0 = keep verbatim)")
    return parser.parse_args(argv)


def load_golden_ids(path: Path) -> set[str]:
    """Task ids of the golden held-out suite (records there are never trained on)."""
    golden_ids: set[str] = set()
    if not path.exists():
        return golden_ids
    for _, record in sc.iter_jsonl(path):
        task_id = record.get("task_id")
        if isinstance(task_id, str) and task_id:
            golden_ids.add(task_id)
    return golden_ids


def content_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def stratified_split(
    records: list[dict[str, Any]], fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into (train, eval), stratified by (domain, task_type, tier).

    Eval quotas use largest-remainder allocation so the global eval share
    approximates `fraction`; members are picked within each stratum by a
    deterministic hash of (seed, task_id). At least one record always stays
    in train.
    """
    try:
        fraction = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid eval fraction: {fraction!r}") from exc
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"eval fraction out of range: {fraction}")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        key = (
            str(record.get("domain")),
            str(record.get("task_type")),
            sc.classify_tier(record),
        )
        groups[key].append(record)

    total = len(records)
    try:
        total_eval = min(int(fraction * total + 0.5), max(0, total - 1))
    except (TypeError, ValueError, OverflowError) as exc:
        # Defensive: fraction and records are validated before this point.
        raise ValueError(f"stratified_split: invalid inputs (fraction={fraction!r}): {exc}") from exc

    quotas: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str, str]]] = []
    for key, group in sorted(groups.items()):
        exact: float | None = None
        try:
            exact = float(fraction * len(group))
            floor_count = int(exact)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"unrepresentable eval quota for {key}: {exact!r}") from exc
        quotas[key] = floor_count
        remainders.append((exact - floor_count, key))

    assigned = sum(quotas.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    changed = True
    while assigned < total_eval and changed:
        changed = False
        for _, key in remainders:
            if assigned >= total_eval:
                break
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                assigned += 1
                changed = True

    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        def order(record: dict[str, Any], seed: int = seed) -> str:
            return hashlib.sha256(f"{seed}|{record['task_id']}".encode()).hexdigest()
        ordered = sorted(group, key=order)
        quota = min(quotas[key], len(group))
        eval_records.extend(ordered[:quota])
        train_records.extend(ordered[quota:])
    return train_records, eval_records


def emit_record(record: dict[str, Any], max_tool_result_chars: int) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "tier": sc.classify_tier(record),
        "domain": record.get("domain"),
        "task_type": record.get("task_type"),
        "difficulty": record.get("difficulty"),
        "messages": sc.record_to_messages(record, max_tool_result_chars),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.eval_fraction < 1:
        print("--eval-fraction must be between 0 and 1", file=sys.stderr)
        return 2
    if args.max_tool_result_chars < 0:
        print("--max-tool-result-chars must be >= 0", file=sys.stderr)
        return 2
    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2
    try:
        raw = [(ln, rec) for ln, rec in sc.iter_jsonl(args.corpus)]
        golden_ids = load_golden_ids(args.golden_eval)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}", file=sys.stderr)
        return 2
    if not raw:
        print(f"corpus is empty: {args.corpus}", file=sys.stderr)
        return 2

    # 1. schema validation
    invalid: list[str] = []
    valid: list[dict[str, Any]] = []
    for line_number, record in raw:
        label = record.get("task_id") or f"line {line_number}"
        errors = sc.validate_record(record, label)
        if errors:
            invalid.extend(errors)
        else:
            valid.append(record)
    if invalid and not args.drop_invalid:
        for error in invalid:
            print(f"SCHEMA VIOLATION: {error}", file=sys.stderr)
        print(f"{len(invalid)} schema violation(s) in {args.corpus}; "
              "rerun with --drop-invalid to quarantine them", file=sys.stderr)
        return 1

    # 2. dedup by task_id (first wins) and by exact content digest
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    duplicate_ids: list[str] = []
    content_duplicates = 0
    deduped: list[dict[str, Any]] = []
    for record in valid:
        task_id = record["task_id"]
        if task_id in seen_ids:
            duplicate_ids.append(task_id)
            continue
        digest = content_digest(record)
        if digest in seen_digests:
            content_duplicates += 1
            continue
        seen_ids.add(task_id)
        seen_digests.add(digest)
        deduped.append(record)

    # 3. golden-suite leakage exclusion
    kept = [r for r in deduped if r["task_id"] not in golden_ids]
    leakage = sorted(r["task_id"] for r in deduped if r["task_id"] in golden_ids)

    # 4. quality tiers
    tier_counts = collections.Counter(sc.classify_tier(r) for r in kept)

    # 5. stratified split
    try:
        train_records, eval_records = stratified_split(kept, args.eval_fraction, args.seed)
    except ValueError as exc:
        print(f"cannot compute stratified split: {exc}", file=sys.stderr)
        return 2

    # 6. render + write
    emitted_train = [emit_record(r, args.max_tool_result_chars) for r in train_records]
    emitted_eval = [emit_record(r, args.max_tool_result_chars) for r in eval_records]
    for record in emitted_train + emitted_eval:
        errors = sc.validate_sft_record(record, record.get("task_id", "?"))
        if errors:
            for error in errors:
                print(f"EMISSION BUG: {error}", file=sys.stderr)
            return 1
    if not emitted_train:
        print("no train records after validation/dedup/leakage exclusion", file=sys.stderr)
        return 1

    train_path = args.out_dir / "sft_train.jsonl"
    eval_path = args.out_dir / "sft_eval.jsonl"
    manifest_path = args.out_dir / "manifest.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_path, emitted_train)
    write_jsonl(eval_path, emitted_eval)

    sizes = sorted(sum(len(m["content"]) for m in r["messages"]) for r in emitted_train + emitted_eval)
    over_budget = sum(1 for size in sizes if size > SEQ_WARN_CHARS)
    train_tiers = collections.Counter(r["tier"] for r in emitted_train)
    eval_tiers = collections.Counter(r["tier"] for r in emitted_eval)

    manifest = {
        "schema": "spec-0001-dataset-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "corpus": str(args.corpus),
            "corpus_sha256": sc.sha256_of_file(args.corpus),
            "golden_eval": str(args.golden_eval),
            "out_dir": str(args.out_dir),
            "eval_fraction": args.eval_fraction,
            "seed": args.seed,
            "drop_invalid": args.drop_invalid,
            "max_tool_result_chars": args.max_tool_result_chars,
        },
        "counts": {
            "corpus_records": len(raw),
            "schema_invalid": len(invalid),
            "duplicate_task_ids": len(duplicate_ids),
            "content_duplicates": content_duplicates,
            "golden_leakage_excluded": len(leakage),
            "kept": len(kept),
            "gold": tier_counts.get("gold", 0),
            "silver": tier_counts.get("silver", 0),
            "train": len(emitted_train),
            "eval": len(emitted_eval),
            "train_tiers": dict(train_tiers),
            "eval_tiers": dict(eval_tiers),
        },
        "leaked_task_ids": leakage,
        "duplicate_task_ids": sorted(set(duplicate_ids)),
        "strata": [],  # filled below from the unique strata table
        "rendered_size": {
            "chars_median": sizes[len(sizes) // 2] if sizes else 0,
            "chars_max": sizes[-1] if sizes else 0,
            "over_seq4096_est": over_budget,
        },
        "files": {
            "train": {"path": str(train_path), "records": len(emitted_train)},
            "eval": {"path": str(eval_path), "records": len(emitted_eval)},
        },
    }
    # proper strata table (unique keys, sorted)
    strata: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in emitted_train:
        key = (record["domain"], record["task_type"], record["tier"])
        strata.setdefault(key, {"domain": key[0], "task_type": key[1], "tier": key[2], "train": 0, "eval": 0})
        strata[key]["train"] += 1
    for record in emitted_eval:
        key = (record["domain"], record["task_type"], record["tier"])
        strata.setdefault(key, {"domain": key[0], "task_type": key[1], "tier": key[2], "train": 0, "eval": 0})
        strata[key]["eval"] += 1
    manifest["strata"] = [strata[key] for key in sorted(strata)]

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print("Dataset build report (spec-0001 section 8)")
    print(f"  corpus: {args.corpus} ({len(raw)} records)")
    print(f"  schema violations: {len(invalid)}"
          + (" (dropped)" if invalid and args.drop_invalid else ""))
    print(f"  duplicates removed: task_id={len(duplicate_ids)} content={content_duplicates}")
    print(f"  golden-suite leakage excluded: {len(leakage)} "
          f"(golden suite: {args.golden_eval}, {len(golden_ids)} task ids)")
    print(f"  quality tiers: gold={tier_counts.get('gold', 0)} silver={tier_counts.get('silver', 0)}")
    print(f"  split (eval fraction {args.eval_fraction}, seed {args.seed}): "
          f"train={len(emitted_train)} eval={len(emitted_eval)}")
    print(f"    train tiers: gold={train_tiers.get('gold', 0)} silver={train_tiers.get('silver', 0)}")
    print(f"    eval tiers:  gold={eval_tiers.get('gold', 0)} silver={eval_tiers.get('silver', 0)}")
    for entry in manifest["strata"]:
        print(f"    stratum {entry['domain']}/{entry['task_type']}/{entry['tier']}: "
              f"train={entry['train']} eval={entry['eval']}")
    print(f"  rendered sizes: median={manifest['rendered_size']['chars_median']} "
          f"max={manifest['rendered_size']['chars_max']} chars")
    if over_budget:
        print(f"  NOTE: {over_budget} record(s) exceed ~{SEQ_WARN_CHARS} chars "
              "(likely over the 4096-token seq budget; consider --max-seq-length 8192 "
              "or --max-tool-result-chars in a rebuild)")
    print(f"  wrote: {train_path} ({len(emitted_train)} records)")
    print(f"  wrote: {eval_path} ({len(emitted_eval)} records)")
    print(f"  wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
