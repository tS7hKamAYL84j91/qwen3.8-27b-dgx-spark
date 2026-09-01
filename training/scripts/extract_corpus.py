#!/usr/bin/env python3
"""Extract a first-pass admin/Quartermaster corpus from redacted pi sessions.

The archive is treated as read-only.  Session snapshots are de-duplicated by
session filename (the filename contains the pi session UUID), with the newest
snapshot winning.  The extractor deliberately emits only the shared spec-0001
record fields; archive paths and raw session metadata are not copied into the
corpus.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ARCHIVE_ROOT = Path("/home/jim/git/coas-archive/records")
CORPUS_OUTPUT = Path("training/corpus/admin_q_corpus.jsonl")
EVAL_OUTPUT = Path("training/eval/admin_q_suite.jsonl")
INVENTORY_OUTPUT = Path("training/corpus/tool_inventory.json")

WORKSPACES = {
    "--home-jim-git-working-notes-executive-office-admin-assistant--": {
        "label": "admin-assistant",
        "domain": "admin",
    },
    "--home-jim-git-coas--": {
        "label": "coas",
        "domain": "quartermaster",
    },
    "--home-jim-git-working-notes-executive-office-executive-assistant--": {
        "label": "executive-assistant",
        "domain": "admin",
    },
}

TASK_TYPES = (
    "status-report",
    "schedule",
    "inventory-update",
    "agent-handoff",
    "doc-format",
    "other",
)

# These patterns are intentionally conservative about ordinary prose, while
# covering the credential formats most likely to occur in shell/tool traces.
SENSITIVE_TOPIC_PROMPT_PATTERNS = (
    re.compile(r"\boauth\b", re.IGNORECASE),
    re.compile(r"\b(?:api|access|refresh|bearer|client)\s+(?:key|token|secret|credential)s?\b", re.IGNORECASE),
    re.compile(r"\b(?:password|passwd|private\s+key|credentials?|secrets?)\b", re.IGNORECASE),
    re.compile(r"\b(?:auth(?:entication|orization)?|token)\.(?:json|env)\b", re.IGNORECASE),
    re.compile(r"\b(?:auth|credential|secret|password|token)[_-](?:file|path|store|script)\b", re.IGNORECASE),
    re.compile(r"\b(?:rotate|refresh|generate|bootstrap|store|manage)\s+(?:the\s+)?(?:api\s+)?tokens?\b", re.IGNORECASE),
    re.compile(r"\btoken\s+scripts?\b", re.IGNORECASE),
    re.compile(r"\b(?:tls|ssl)\s+cert(?:ificate)?s?\b|\bcertificates?\b", re.IGNORECASE),
    re.compile(r"\bweb[- ]access\b", re.IGNORECASE),
)

BOILERPLATE_PROMPT_PATTERNS = (
    re.compile(r"\bfirst\s*:\s*call\s+set_name\b", re.IGNORECASE),
    re.compile(r"\brun\s+the\s+rest\s+of\s+your\s+startup\s+checklist\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+new\s+messages?\b.*\bmessage_read\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bwe\s+seem\s+to\s+have\s+stopped\b", re.IGNORECASE),
    re.compile(r"\bwe\s+(?:seem|appear)\s+to\s+have\s+(?:stopped|paused)\b", re.IGNORECASE),
    re.compile(r"^\s*continue\s+(?:the\s+)?(?:session|workspace\s+task)\b", re.IGNORECASE),
    re.compile(r"\b(?:continue|resume|pick\s+up\s+where\s+we\s+left\s+off)\b.{0,80}\b(?:session|conversation|workspace\s+task)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:again|ok(?:ay)?|hmm|hello|hi)[\s.!?]*$", re.IGNORECASE),
)

SENSITIVE_CONTENT_PATTERN = re.compile(
    r"(?:oauth|token|credential|password|api[-_]?key|secret)",
    re.IGNORECASE,
)

SECRET_PATTERNS = (
    (re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.I | re.S), "[[REDACTED_SECRET]]"),
    (re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{12,}\b"), "[[REDACTED_SECRET]]"),
    (re.compile(r"\b(?:ghp|gho|ghs|ghr|github_pat)_[A-Za-z0-9_]{12,}\b"), "[[REDACTED_SECRET]]"),
    (re.compile(r"\b(?:xox[baprs])-[-A-Za-z0-9]{12,}\b"), "[[REDACTED_SECRET]]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[[REDACTED_SECRET]]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "[[REDACTED_SECRET]]"),
    (re.compile(r"\b(?:npm|pypi)-[A-Za-z0-9_-]{12,}\b", re.I), "[[REDACTED_SECRET]]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"), r"\1[[REDACTED_SECRET]]"),
    (re.compile(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@"), r"\1[[REDACTED_SECRET]]@"),
    (re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?key|secret|password|passwd|token|authorization)\b\s*[:=]\s*[\"']?)[^\s,\"'}]+"), r"\1[[REDACTED_SECRET]]"),
)


def redact(value: Any) -> Any:
    """Redact secret-like strings recursively without changing JSON shape."""
    if isinstance(value, str):
        result = value.replace("\x00", "")
        for pattern, replacement in SECRET_PATTERNS:
            result = pattern.sub(replacement, result)
        return result
    if isinstance(value, dict):
        return {str(key): redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def bound_text(value: str, limit: int) -> str:
    value = redact(value)
    if len(value) <= limit:
        return value
    if limit < 80:
        return value[:limit]
    head = (limit - 60) // 2
    tail = limit - head - 60
    return value[:head] + "\n...[TRUNCATED]...\n" + value[-tail:]


def content_text(content: Any) -> str:
    """Get visible text from pi's string or content[] representation."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            chunks.append(item["text"])
    return "\n".join(chunks)


def parse_arguments(arguments: Any) -> dict[str, Any]:
    """Normalize pi tool arguments to the schema's object-shaped ``args``."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return redact(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return {"raw": bound_text(arguments, 4000)}
        if isinstance(parsed, dict):
            return redact(parsed)
        return {"value": redact(parsed)}
    return {"value": redact(arguments)}


def iter_message_content(messages: Iterable[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]], int, int, collections.Counter[str]]:
    """Return user prompts, trajectory events, tool-call count, and text size."""
    prompts: list[str] = []
    events: list[dict[str, Any]] = []
    tool_calls = 0
    text_size = 0
    tool_counts: collections.Counter[str] = collections.Counter()

    for entry in messages:
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            text = content_text(content).strip()
            if text:
                prompts.append(text)
                text_size += len(text)

        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "toolCall":
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                events.append(
                    {
                        "role": "assistant",
                        "tool_call": {
                            "name": bound_text(name.strip(), 200),
                            "args": parse_arguments(item.get("arguments")),
                        },
                    }
                )
                tool_calls += 1
                tool_counts[name.strip()] += 1
            elif role == "toolResult" and item_type == "text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    events.append({"role": "tool", "result": bound_text(text, 4000)})
                    text_size += len(text)

    return prompts, events, tool_calls, text_size, tool_counts


def read_session(path: Path, max_events: int, max_text: int) -> tuple[dict[str, Any], collections.Counter[str]]:
    """Parse one JSONL session, skipping archive redaction marker lines."""
    messages: list[dict[str, Any]] = []
    stats: collections.Counter[str] = collections.Counter()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                stats["lines"] += 1
                if re.search(r"quartermaster", line, re.IGNORECASE):
                    stats["quartermaster_match"] = 1
                try:
                    entry = json.loads(line)
                except (TypeError, ValueError):
                    stats["invalid_lines"] += 1
                    continue
                if not isinstance(entry, dict):
                    continue
                messages.append(entry)
    except OSError:
        stats["read_errors"] += 1

    prompts, events, tool_calls, text_size, tool_counts = iter_message_content(messages)
    stats["tool_calls"] = tool_calls
    stats["text_chars"] = text_size
    stats["messages"] = len(messages)
    if prompts:
        stats["sessions_with_prompt"] = 1
    if events:
        stats["sessions_with_trajectory"] = 1

    # Keep the beginning and end: the beginning usually contains setup/tool
    # selection and the end contains the completion/final tool result.
    if len(events) > max_events:
        first_count = max_events // 2
        events = events[:first_count] + [{"role": "tool", "result": "...[trajectory truncated]..."}] + events[-(max_events - first_count - 1):]
        stats["truncated_trajectories"] = 1

    assistant_texts: list[str] = []
    for entry in messages:
        message = entry.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = content_text(message.get("content")).strip()
        if text:
            assistant_texts.append(text)

    # A session is one task unit.  Use its first user turn rather than
    # concatenating later turns, which may belong to unrelated work.
    prompt = prompts[0].strip() if prompts else "Continue the workspace task captured in this session."
    if not prompt:
        prompt = "Continue the workspace task captured in this session."
    final_output = assistant_texts[-1] if assistant_texts else "No final textual response was recorded."
    if assistant_texts:
        stats["sessions_with_final_text"] = 1

    return {
        "prompt": bound_text(prompt, max_text),
        "events": events,
        "final_output": bound_text(final_output, max_text),
        "tool_names": sorted(tool_counts),
        "tool_counts": dict(tool_counts),
    }, stats


def is_boilerplate_prompt(prompt: str) -> bool:
    stripped = " ".join(prompt.split())
    if len(stripped) < 40:
        return True
    return any(pattern.search(stripped) for pattern in BOILERPLATE_PROMPT_PATTERNS)


def contains_sensitive_content(value: Any) -> bool:
    """Check serialized corpus values, including nested tool/context fields."""
    if isinstance(value, str):
        return bool(SENSITIVE_CONTENT_PATTERN.search(value))
    if isinstance(value, dict):
        return any(contains_sensitive_content(key) or contains_sensitive_content(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_sensitive_content(item) for item in value)
    return False


def is_sensitive_topic_prompt(prompt: str) -> bool:
    return contains_sensitive_content(prompt) or any(pattern.search(prompt) for pattern in SENSITIVE_TOPIC_PROMPT_PATTERNS)


def has_real_output(parsed: dict[str, Any]) -> bool:
    output = parsed["final_output"].strip()
    if len(output) < 40:
        return False
    return output != "No final textual response was recorded."


def classify_task(prompt: str, tool_names: list[str], final_output: str) -> str:
    haystack = (prompt + " " + final_output + " " + " ".join(tool_names)).lower()
    if any(word in haystack for word in ("inventory", "stock", "crate", "chain of custody", "warehouse")):
        return "inventory-update"
    if any(word in haystack for word in ("schedule", "calendar", "meeting", "deadline", "due date")):
        return "schedule"
    if any(word in haystack for word in ("handoff", "hand-off", "delegate", "agent_send", "message_send", "spawn_agent", "agent hand")):
        return "agent-handoff"
    if any(word in haystack for word in ("status", "health", "kanban", "check-in", "check in", "progress report", "agent_status")):
        return "status-report"
    return "other"


def difficulty(event_count: int) -> str:
    """Classify by reference-trajectory steps, per the extraction contract."""
    if event_count < 5:
        return "S"
    if event_count < 20:
        return "M"
    return "L"


def make_record(path: Path, workspace: str, parsed: dict[str, Any], stats: Mapping[str, int]) -> dict[str, Any]:
    info = WORKSPACES[workspace]
    label = info["label"]
    filename = path.name
    stem = filename[:-6] if filename.endswith(".jsonl") else filename
    uuid_part = stem.split("_")[-1]
    uuid_part = re.sub(r"[^A-Za-z0-9-]", "", uuid_part) or hashlib.sha256(filename.encode()).hexdigest()[:12]
    prefix = "qm" if info["domain"] == "quartermaster" else "adm"
    task_id = f"{prefix}-{uuid_part}"
    task_type = classify_task(parsed["prompt"], parsed["tool_names"], parsed["final_output"])

    if task_type in ("inventory-update", "agent-handoff"):
        validator_kind = "state-transition"
    else:
        validator_kind = "template-match"

    return {
        "task_id": task_id,
        "domain": info["domain"],
        "task_type": task_type,
        "prompt": parsed["prompt"],
        "context": {
            "tools": parsed["tool_names"],
            "prior_state": {
                "workspace": label,
                "session_filename": filename,
            },
        },
        "reference_trajectory": parsed["events"],
        "final_output": parsed["final_output"],
        "validator": {
            "kind": validator_kind,
            "spec": {"task_type": task_type, "source": "spec-0001-first-pass"},
        },
        "difficulty": difficulty(len(parsed["events"])),
        "source": "coas-trace",
        "needs_validation": True,
    }


def discover_sessions(archive_root: Path) -> tuple[dict[str, tuple[str, Path]], int]:
    """Find snapshots and return filename -> (workspace, newest path)."""
    candidates: dict[str, list[tuple[str, int, str, Path]]] = collections.defaultdict(list)
    discovered = 0
    if not archive_root.exists():
        return {}, 0
    for path in archive_root.rglob("*.jsonl"):
        parent = path.parent
        if parent.parent.name != "sessions" or parent.name not in WORKSPACES:
            continue
        discovered += 1
        # Snapshot directory names are UTC-sortable.  Prefer the newest copy;
        # within one snapshot prefer the largest copy, then a stable path.
        snapshot = parent.parent.parent.name
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        candidates[path.name].append((snapshot, size, str(path), path))

    selected: dict[str, tuple[str, Path]] = {}
    for filename, paths in candidates.items():
        _, _, _, path = max(paths, key=lambda item: (item[0], item[1], item[2]))
        selected[filename] = (path.parent.name, path)
    return selected, discovered


def choose_eval(records: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use a stable task-id hash threshold for the held-out suite."""
    eval_records: list[dict[str, Any]] = []
    for record in records:
        digest = hashlib.sha256(record["task_id"].encode()).hexdigest()
        try:
            score = int(digest[:16], 16) / float(1 << 64)
        except (OverflowError, ValueError):
            continue
        if score < fraction:
            eval_records.append(record)
    eval_ids = {record["task_id"] for record in eval_records}
    corpus = [record for record in records if record["task_id"] not in eval_ids]
    return sorted(corpus, key=lambda record: record["task_id"]), sorted(eval_records, key=lambda record: record["task_id"])


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(redact(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_inventory(path: Path, inventory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(redact(inventory), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract deduplicated spec-0001 admin/Quartermaster records from coas-archive pi sessions."
    )
    parser.add_argument("--archive", "--archive-root", dest="archive", type=Path, default=ARCHIVE_ROOT, help=f"Archive records root (default: {ARCHIVE_ROOT})")
    parser.add_argument("--out-dir", type=Path, default=CORPUS_OUTPUT.parent, help=f"Corpus output directory (default: {CORPUS_OUTPUT.parent})")
    parser.add_argument("--eval-out", "--eval-output", dest="eval_out", type=Path, default=EVAL_OUTPUT, help=f"Held-out eval JSONL output (default: {EVAL_OUTPUT})")
    parser.add_argument("--eval-fraction", type=float, default=0.05, help="Held-out hash fraction (default: 0.05)")
    parser.add_argument("--max-trajectory-events", type=int, default=256, help="Maximum reference trajectory events per record (default: 256)")
    parser.add_argument("--max-text-chars", type=int, default=12000, help="Maximum prompt/final-output characters (default: 12000)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 < args.eval_fraction < 1:
        parser.error("--eval-fraction must be between 0 and 1")
    if args.max_trajectory_events < 3 or args.max_text_chars < 80:
        parser.error("trajectory limit must be >= 3 and text limit must be >= 80")

    corpus_output = args.out_dir / "admin_q_corpus.jsonl"
    inventory_output = args.out_dir / "tool_inventory.json"
    selected, discovered = discover_sessions(args.archive)
    if not selected:
        print(f"No target session files found below {args.archive}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    inventory: dict[str, dict[str, Any]] = {
        info["label"]: {
            "workspace_dir": workspace,
            "domain": info["domain"],
            "session_files": 0,
            "tool_calls": 0,
            "tools": {},
        }
        for workspace, info in WORKSPACES.items()
    }
    parse_totals = collections.Counter()
    for filename in sorted(selected):
        workspace, path = selected[filename]
        parsed, stats = read_session(path, args.max_trajectory_events, args.max_text_chars)
        if stats["lines"] < 10:
            parse_totals["skipped_short_sessions"] += 1
            continue
        if workspace == "--home-jim-git-coas--" and not stats["quartermaster_match"]:
            parse_totals["skipped_non_quartermaster"] += 1
            continue
        if is_boilerplate_prompt(parsed["prompt"]):
            parse_totals["skipped_boilerplate_prompts"] += 1
            continue
        if is_sensitive_topic_prompt(parsed["prompt"]):
            parse_totals["skipped_sensitive_topics"] += 1
            continue
        if not has_real_output(parsed):
            parse_totals["skipped_placeholder_outputs"] += 1
            continue
        if stats["tool_calls"] == 0 and len(parsed["final_output"].strip()) < 80:
            parse_totals["skipped_no_tool_short_output"] += 1
            continue
        record = make_record(path, workspace, parsed, stats)
        if (
            contains_sensitive_content(record["final_output"])
            or contains_sensitive_content(record["reference_trajectory"])
            or contains_sensitive_content(record["context"])
        ):
            parse_totals["skipped_sensitive_topics"] += 1
            continue
        records.append(record)
        label = WORKSPACES[workspace]["label"]
        bucket = inventory[label]
        bucket["session_files"] += 1
        bucket["tool_calls"] += stats["tool_calls"]
        for tool_name, count in parsed["tool_counts"].items():
            bucket["tools"][tool_name] = bucket["tools"].get(tool_name, 0) + count
        parse_totals.update(stats)

    corpus, eval_records = choose_eval(records, args.eval_fraction)
    for bucket in inventory.values():
        bucket["tools"] = dict(sorted(bucket["tools"].items()))
    inventory_document = {
        "schema": "spec-0001",
        "source": "coas-trace",
        "dedupe_key": "session_filename",
        "discovered_snapshot_files": discovered,
        "unique_session_files": len(selected),
        "emitted_records": len(records),
        "workspaces": inventory,
    }

    write_jsonl(corpus_output, corpus)
    write_jsonl(args.eval_out, eval_records)
    write_inventory(inventory_output, inventory_document)

    print("Corpus report")
    print(f"  archive: {args.archive}")
    print(f"  snapshot files discovered: {discovered}")
    print(f"  unique session files considered: {len(selected)} (duplicates removed: {discovered - len(selected)})")
    print(f"  records: {len(records)} (corpus={len(corpus)}, eval={len(eval_records)})")
    rejection_count = sum(
        parse_totals[key]
        for key in (
            "skipped_short_sessions",
            "skipped_non_quartermaster",
            "skipped_boilerplate_prompts",
            "skipped_sensitive_topics",
            "skipped_placeholder_outputs",
            "skipped_no_tool_short_output",
        )
    )
    print(f"  rejected records: {rejection_count}")
    print(
        "  rejection reasons: "
        f"short={parse_totals['skipped_short_sessions']} "
        f"non-quartermaster={parse_totals['skipped_non_quartermaster']} "
        f"boilerplate={parse_totals['skipped_boilerplate_prompts']} "
        f"sensitive-topic={parse_totals['skipped_sensitive_topics']} "
        f"placeholder/short-output={parse_totals['skipped_placeholder_outputs']} "
        f"no-tool-short-output={parse_totals['skipped_no_tool_short_output']}"
    )
    print(f"  invalid JSONL lines skipped: {parse_totals['invalid_lines']}")
    denominator = len(records) or 1
    print(f"  sessions with user prompt: {parse_totals['sessions_with_prompt']} ({parse_totals['sessions_with_prompt'] / denominator:.1%})")
    print(f"  sessions with final assistant text: {parse_totals['sessions_with_final_text']} ({parse_totals['sessions_with_final_text'] / denominator:.1%})")
    print(f"  sessions with tool trajectories: {parse_totals['sessions_with_trajectory']} ({parse_totals['sessions_with_trajectory'] / denominator:.1%})")
    print(f"  tool calls observed: {parse_totals['tool_calls']} (mean/session={parse_totals['tool_calls'] / denominator:.1f})")
    task_counts = collections.Counter(record["task_type"] for record in records)
    difficulty_counts = collections.Counter(record["difficulty"] for record in records)
    print(f"  task types: {dict(sorted(task_counts.items()))}")
    print(f"  difficulty: {dict(sorted(difficulty_counts.items()))}")
    print("  per-workspace:")
    for label in sorted(inventory):
        bucket = inventory[label]
        corpus_count = sum(1 for record in corpus if record["context"]["prior_state"]["workspace"] == label)
        eval_count = sum(1 for record in eval_records if record["context"]["prior_state"]["workspace"] == label)
        top_tools = sorted(bucket["tools"].items(), key=lambda item: (-item[1], item[0]))[:15]
        print(f"    {label}: sessions={bucket['session_files']} corpus={corpus_count} eval={eval_count} tool_calls={bucket['tool_calls']}")
        print(f"      top-15 tools: {', '.join(f'{name}={count}' for name, count in top_tools)}")
    print("  2 sample records (abbreviated trajectories):")
    candidates = [
        record for record in records
        if record["prompt"] != "Continue the workspace task captured in this session."
        and record["final_output"] != "No final textual response was recorded."
        and record["reference_trajectory"]
    ]
    samples = (candidates + records)[:2]
    for record in samples:
        sample = dict(record)
        sample["prompt"] = bound_text(record["prompt"], 240)
        sample["final_output"] = bound_text(record["final_output"], 240)
        sample_events: list[dict[str, Any]] = []
        for event in record["reference_trajectory"][:4]:
            if event.get("role") == "tool":
                sample_events.append({"role": "tool", "result": bound_text(event.get("result", ""), 240)})
            else:
                call = event.get("tool_call", {})
                sample_events.append({
                    "role": "assistant",
                    "tool_call": {
                        "name": call.get("name", ""),
                        "args": bound_text(json.dumps(call.get("args", {}), ensure_ascii=False), 240),
                    },
                })
        sample["reference_trajectory"] = sample_events
        print("    " + json.dumps(redact(sample), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
