#!/usr/bin/env python3
"""Validator-based eval of a served checkpoint on the golden admin/Q suite.

Spec 0001 section 6 harness: scores tool-call validity, end-to-end task
completion, and template fidelity per record using the shared record
validators (spec_common), then runs the serving perf check by invoking
benchmarks/bench.py and comparing short-decode tok/s against the ADR-0002
promoted baseline.

The checkpoint must be SERVED (e.g. the resident SGLang runtime exposing an
OpenAI-compatible /v1/chat/completions). This script is an HTTP client only:
it never touches the GPU directly. Multi-turn rollouts replay the reference
trajectory's recorded tool results as a deterministic stand-in environment.

Run on the host (needs benchmarks/bench.py in the repo tree):

    python3 training/scripts/run_eval.py --checkpoint qwen38-sft-adapter-r1

Exit codes: 0 all gates pass; 1 gates failed (or perf unavailable);
2 harness error (suite/serving/usage).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Sibling module: resolves at runtime because sys.path[0] is this
# script's directory when invoked per the spec-0001 section-8 runbook
# (python3 .../scripts/<name>.py).
import spec_common as sc  # pyright: ignore[reportMissingImports]

TRAINING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_DIR.parent
DEFAULT_SUITE = TRAINING_DIR / "eval" / "admin_q_suite.jsonl"
DEFAULT_RESULTS_DIR = TRAINING_DIR / "eval" / "results"
DEFAULT_BASE_URL = "http://127.0.0.1:18083/v1"
DEFAULT_MODEL = "qwen3.8"
DEFAULT_BENCH = REPO_ROOT / "benchmarks" / "bench.py"

# Spec 0001 section 2 capability targets. The completion target is defined
# relative to a validated cloud-teacher baseline on the same suite; until that
# baseline is recorded we gate on the absolute floor below.
TARGETS = {
    "tool_call_fidelity_pct_min": 98.0,
    "template_fidelity_pct_min": 98.0,
    "completion_rate_pct_min": 90.0,
}
# ADR-0002: promoted short-decode baseline for the served Qwen3.8-27B profile.
DEFAULT_BASELINE_DECODE_TPS = 22.25
DEFAULT_PERF_TOLERANCE = 0.05

_EXCERPT_CHARS = 400


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a served checkpoint on the golden admin/Q suite (spec 0001 section 6)."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint label or path being evaluated (used in the report)")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE,
                        help=f"golden held-out JSONL (default: {DEFAULT_SUITE})")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help=f"OpenAI-compatible chat endpoint (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"served model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-turns", type=int, default=6,
                        help="max rollout turns per record before the model must answer (default: 6)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="max generated tokens per turn (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="generation temperature (default: 0 greedy, bench convention)")
    parser.add_argument("--timeout", type=float, default=600,
                        help="per-request HTTP timeout in seconds (default: 600)")
    parser.add_argument("--skip-perf", action="store_true",
                        help="skip the bench.py decode tok/s perf check")
    parser.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH,
                        help=f"benchmarks/bench.py (default: {DEFAULT_BENCH})")
    parser.add_argument("--bench-backend", choices=["sglang", "ollama"], default="sglang",
                        help="bench.py backend (default: sglang)")
    parser.add_argument("--bench-tests", type=str, default="short",
                        help="bench.py tests (default: short; decode tok/s gate uses 'short')")
    parser.add_argument("--bench-runs", type=int, default=2,
                        help="bench.py runs per test (default: 2)")
    parser.add_argument("--baseline-decode-tps", type=float, default=DEFAULT_BASELINE_DECODE_TPS,
                        help=f"ADR-0002 short-decode baseline tok/s (default: {DEFAULT_BASELINE_DECODE_TPS})")
    parser.add_argument("--perf-tolerance", type=float, default=DEFAULT_PERF_TOLERANCE,
                        help="allowed decode tok/s deviation from baseline (default: 0.05 = within 5%%)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON path (default: training/eval/results/eval_<label>_<ts>.json)")
    return parser.parse_args(argv)


def chat_completion(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    """One chat completion; raises URLError/HTTPError/TimeoutError upward."""
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        try:
            body = json.load(response)
        except ValueError as exc:
            raise RuntimeError(f"malformed JSON in chat completion response: {exc}") from exc
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError("chat completion response has no choices")
    return choices[0].get("message") or {}


def rollout_record(args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    """Generate on one record, replaying reference tool results turn by turn."""
    messages = sc.generation_context(record)
    replay = sc.replay_tool_results(record)
    calls: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    final_output = ""

    for turn_number in range(1, args.max_turns + 1):
        message = chat_completion(args, messages)
        raw_content = message.get("content")
        content = raw_content if isinstance(raw_content, str) else ""
        native_calls = sc.tool_calls_from_api_message(message)
        text_calls = sc.parse_tool_calls(content)
        turn_calls = sc.merge_tool_calls(native_calls, text_calls)
        turns.append({
            "turn": turn_number,
            "content_excerpt": content[:_EXCERPT_CHARS],
            "tool_calls": [
                {"name": c.get("name"), "args": c.get("args")} for c in turn_calls
            ],
        })
        if not turn_calls:
            final_output = content
            break
        calls.extend(turn_calls)
        assistant_content = content if content.strip() else "\n\n".join(
            sc.format_tool_call(c) for c in turn_calls
        )
        messages.append({"role": "assistant", "content": str(assistant_content)})
        results: list[str] = []
        for _ in turn_calls:
            results.append(
                replay.pop(0)
                if replay
                else "Error: eval replay has no reference result for this step; "
                     "stop calling tools and produce the final output."
            )
        messages.append({"role": "tool", "content": "\n\n".join(results)})
    else:
        final_output = ""  # exhausted turns without a final answer

    return {
        "task_id": record.get("task_id"),
        "turns_taken": len(turns),
        "turns": turns,
        "calls": calls,
        "final_output": final_output,
        "final_output_excerpt": final_output[:_EXCERPT_CHARS],
    }


def score_record(record: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    """Apply the record validators to one rollout."""
    available = record.get("context", {}).get("tools") or []
    call_results: list[dict[str, Any]] = []
    for call in rollout["calls"]:
        reasons = sc.validate_tool_call(call.get("name"), call.get("args"), available)
        call_results.append({
            "name": call.get("name"),
            "args": call.get("args"),
            "valid": not reasons,
            "reasons": reasons,
        })
    kind = (record.get("validator") or {}).get("kind")
    output = rollout["final_output"]
    if kind == "template-match":
        validator_ok, validator_reasons = sc.check_template(record, output)
    elif kind == "state-transition":
        validator_ok, validator_reasons = sc.check_state_transition(record, call_results, output)
    elif kind == "json-schema":
        validator_ok, validator_reasons = sc.check_json_schema(record, output)
    else:
        validator_ok, validator_reasons = None, [f"unknown validator kind {kind!r}"]
    completed, completion_reasons = sc.check_completion(
        record, output, call_results, validator_ok, validator_reasons
    )
    return {
        "task_id": record.get("task_id"),
        "domain": record.get("domain"),
        "task_type": record.get("task_type"),
        "validator_kind": kind,
        "reference_tool_calls": sc.count_tool_calls(record),
        "turns_taken": rollout["turns_taken"],
        "emitted_calls": len(call_results),
        "valid_calls": sum(1 for c in call_results if c["valid"]),
        "call_results": call_results,
        "validator_ok": validator_ok,
        "validator_reasons": validator_reasons,
        "completed": completed,
        "completion_reasons": completion_reasons,
        "final_output_excerpt": rollout["final_output_excerpt"],
        "turns": rollout["turns"],
    }


def aggregate(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    total_calls = sum(t["emitted_calls"] for t in per_task)
    valid_calls = sum(t["valid_calls"] for t in per_task)
    template_tasks = [t for t in per_task if t["validator_kind"] == "template-match"]
    template_pass = sum(
        1 for t in template_tasks
        if t["validator_ok"] is not None and t["validator_ok"]  # None = n/a, False = failed
    )
    completed = sum(1 for t in per_task if t["completed"])
    return {
        "records": len(per_task),
        "tool_calls_total": total_calls,
        "tool_calls_valid": valid_calls,
        "tool_call_fidelity_pct": (100.0 * valid_calls / total_calls) if total_calls else None,
        "records_with_no_calls": sum(1 for t in per_task if t["emitted_calls"] == 0),
        "template_match_records": len(template_tasks),
        "template_match_pass": template_pass,
        "template_fidelity_pct": (100.0 * template_pass / len(template_tasks)) if template_tasks else None,
        "state_transition_records": sum(1 for t in per_task if t["validator_kind"] == "state-transition"),
        "completion_pass": completed,
        "completion_rate_pct": (100.0 * completed / len(per_task)) if per_task else None,
    }


def run_perf(args: argparse.Namespace) -> dict[str, Any]:
    """Perf check hook: invoke benchmarks/bench.py for decode tok/s."""
    perf: dict[str, Any] = {
        "status": "skipped",
        "backend": args.bench_backend,
        "tests": args.bench_tests,
        "runs": args.bench_runs,
        "baseline_decode_tps": args.baseline_decode_tps,
        "tolerance": args.perf_tolerance,
        "decode_tps_median": None,
        "within_baseline": None,
    }
    if not args.bench_path.exists():
        perf["status"] = "unavailable"
        perf["error"] = f"bench.py not found at {args.bench_path}"
        return perf
    try:
        with tempfile.TemporaryDirectory(prefix="qwen38-eval-bench-") as tmp_dir:
            out_path = str(Path(tmp_dir) / "bench.json")
            command = [
                sys.executable, str(args.bench_path),
                "--backend", args.bench_backend,
                "--tests", args.bench_tests,
                "--runs", str(args.bench_runs),
                "--out", out_path,
            ]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                perf["status"] = "unavailable"
                perf["error"] = (proc.stderr or proc.stdout)[-800:]
                return perf
            with open(out_path, "r", encoding="utf-8") as stream:
                results = json.load(stream)
        medians: dict[str, float] = {}
        for test, entries in results.items():
            decodes = [entry.get("decode_tps") for entry in entries if entry.get("decode_tps")]
            if decodes:
                medians[test] = statistics.median(decodes)
        perf["status"] = "ok"
        perf["decode_tps_medians"] = medians
        if "short" in medians:
            median = medians["short"]
            perf["decode_tps_median"] = median
            lower = args.baseline_decode_tps * (1 - args.perf_tolerance)
            upper = args.baseline_decode_tps * (1 + args.perf_tolerance)
            perf["within_baseline"] = bool(lower <= median <= upper)
        else:
            perf["error"] = "no short-test decode tok/s in bench results"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        perf["status"] = "unavailable"
        perf["error"] = str(exc)
    return perf


def compute_gates(metrics: dict[str, Any], perf: dict[str, Any] | None) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    fidelity = metrics["tool_call_fidelity_pct"]
    gates["tool_call_fidelity"] = {
        "value": fidelity,
        "target": ">= 98%",
        "pass": fidelity is not None and fidelity >= TARGETS["tool_call_fidelity_pct_min"],
        "note": None if fidelity is not None else "no tool calls emitted",
    }
    template = metrics["template_fidelity_pct"]
    gates["template_fidelity"] = {
        "value": template,
        "target": ">= 98%",
        "pass": template is None or template >= TARGETS["template_fidelity_pct_min"],
        "note": None if template is not None else "no template-match records in suite",
    }
    completion = metrics["completion_rate_pct"]
    gates["completion"] = {
        "value": completion,
        "target": ">= 90% (absolute floor; spec target is >= 90% of a validated "
                  "cloud-teacher baseline on the same suite)",
        "pass": completion is not None and completion >= TARGETS["completion_rate_pct_min"],
    }
    if perf is None:
        gates["perf"] = {"status": "skipped", "pass": True}
    elif perf.get("status") == "ok":
        gates["perf"] = {
            "status": "ok",
            "decode_tps_median": perf.get("decode_tps_median"),
            "baseline": perf.get("baseline_decode_tps"),
            "pass": bool(perf.get("within_baseline")),  # None (not measured) counts as failed
        }
    else:
        gates["perf"] = {"status": perf.get("status"), "error": perf.get("error"), "pass": False}
    return gates


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_turns < 1 or args.max_tokens < 16 or args.timeout <= 0:
        print("invalid --max-turns/--max-tokens/--timeout", file=sys.stderr)
        return 2
    if args.baseline_decode_tps <= 0:
        print("--baseline-decode-tps must be positive", file=sys.stderr)
        return 2
    if not args.suite.exists():
        print(f"golden suite not found: {args.suite}", file=sys.stderr)
        return 2

    # Suite must be schema-clean before anything runs.
    suite_records: list[dict[str, Any]] = []
    try:
        for line_number, record in sc.iter_jsonl(args.suite):
            errors = sc.validate_record(record, record.get("task_id") or f"line {line_number}")
            if errors:
                for error in errors:
                    print(f"GOLDEN SUITE INVALID: {error}", file=sys.stderr)
                return 2
            suite_records.append(record)
    except (OSError, ValueError) as exc:
        print(f"cannot read suite: {exc}", file=sys.stderr)
        return 2
    if not suite_records:
        print("golden suite is empty", file=sys.stderr)
        return 2

    per_task: list[dict[str, Any]] = []
    endpoint_ok = False
    harness_errors: list[str] = []
    for record in suite_records:
        try:
            rollout = rollout_record(args, record)
            endpoint_ok = True
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError, ValueError) as exc:
            harness_errors.append(f"{record.get('task_id')}: generation failed: {exc}")
            continue
        per_task.append(score_record(record, rollout))
        task = per_task[-1]
        print(f"[{record.get('task_id')}] turns={task['turns_taken']} "
              f"calls={task['emitted_calls']} (valid={task['valid_calls']}) "
              f"validator={task['validator_kind']}:{task['validator_ok']} "
              f"completed={task['completed']}", flush=True)
    if not endpoint_ok:
        for error in harness_errors:
            print(f"HARNESS ERROR: {error}", file=sys.stderr)
        print(f"no records scored; is the checkpoint served at {args.base_url} "
              f"as model {args.model}?", file=sys.stderr)
        return 2

    metrics = aggregate(per_task)
    perf = None if args.skip_perf else run_perf(args)
    if perf is not None and perf.get("status") != "ok":
        print(f"PERF CHECK: {perf.get('status')}: {perf.get('error')}", file=sys.stderr)
    gates = compute_gates(metrics, perf)
    overall_pass = all(gate.get("pass") for gate in gates.values())

    label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.checkpoint)[:80]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or (DEFAULT_RESULTS_DIR / f"eval_{label}_{timestamp}.json")
    document = {
        "schema": "spec-0001-eval-v1",
        "checkpoint": args.checkpoint,
        "suite": str(args.suite),
        "base_url": args.base_url,
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": TARGETS,
        "metrics": metrics,
        "gates": gates,
        "overall_pass": overall_pass,
        "perf": perf,
        "harness_errors": harness_errors,
        "per_task": per_task,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")

    print("\nEval summary (spec 0001 section 2 targets)")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  suite: {args.suite} ({metrics['records']} records)")
    fidelity = metrics["tool_call_fidelity_pct"]
    print(f"  tool-call fidelity: {fidelity if fidelity is not None else 'n/a'}% "
          f"(gate: >= {TARGETS['tool_call_fidelity_pct_min']:.0f}%; "
          f"{metrics['tool_calls_valid']}/{metrics['tool_calls_total']} calls valid, "
          f"{metrics['records_with_no_calls']} record(s) with no calls)")
    template = metrics["template_fidelity_pct"]
    print(f"  template fidelity: {template if template is not None else 'n/a'}% "
          f"(gate: >= {TARGETS['template_fidelity_pct_min']:.0f}%; "
          f"{metrics['template_match_pass']}/{metrics['template_match_records']} template-match records)")
    completion = metrics["completion_rate_pct"]
    print(f"  completion rate: {completion if completion is not None else 'n/a'}% "
          f"(gate: >= {TARGETS['completion_rate_pct_min']:.0f}% absolute floor; "
          f"spec: >= 90% of a validated cloud-teacher baseline)")
    if perf is None:
        print("  perf: skipped (--skip-perf)")
    elif perf.get("status") == "ok":
        print(f"  perf: short decode {perf.get('decode_tps_median')} tok/s vs baseline "
              f"{perf.get('baseline_decode_tps')} tok/s -> "
              f"{'within' if perf.get('within_baseline') else 'OUTSIDE'} tolerance "
              f"{perf.get('tolerance'):.0%}")
    else:
        print(f"  perf: {perf.get('status')} ({perf.get('error')})")
    print(f"  overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  report: {out_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
