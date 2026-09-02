#!/usr/bin/env python3
"""Shared spec-0001 record schema, rendering, and validator logic.

Used by build_dataset.py (corpus schema validation, quality tiers, SFT
rendering), train_sft_qlora.py (dataset loading and validation), and
run_eval.py (the record validators from spec 0001 sections 5 and 6:
tool-call schema check, template fidelity, state-transition and completion
checks).

The validator minimum set required by spec 0001 section 5 lives here as pure
functions so it can be reused for training-data filtering (spec 0002 Stage
1), reward functions (spec 0002 Stage 2), and the eval harness.

Pure stdlib only; safe to import on the host with no GPU stack installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

# --- spec 0001 section 5 schema --------------------------------------------

DOMAINS = ("admin", "quartermaster")
# "other" is the extractor's catch-all bucket (scripts/extract_corpus.py);
# it extends the spec's task_type enum and is accepted here.
TASK_TYPES = (
    "status-report",
    "schedule",
    "inventory-update",
    "agent-handoff",
    "doc-format",
    "other",
)
DIFFICULTIES = ("S", "M", "L")
SOURCES = ("coas-trace", "teacher-generated", "hand-written")
VALIDATOR_KINDS = ("json-schema", "state-transition", "template-match")
TRAJECTORY_ROLES = ("assistant", "tool")

# Quality tiers: gold = the reference trajectory contains at least
# GOLD_MIN_TOOL_CALLS tool calls AND the final_output is at least
# GOLD_MIN_FINAL_OUTPUT_CHARS characters; silver = the rest.
GOLD_MIN_TOOL_CALLS = 2
GOLD_MIN_FINAL_OUTPUT_CHARS = 80

# --- office tool argument schemas ------------------------------------------
# Lightweight type schemas for the office tools observed in the corpus
# (training/corpus/tool_inventory.json), derived from the actual argument
# shapes in the corpus traces. Validation = required keys present with the
# right JSON type, optional keys type-checked when present. Tools without an
# entry here fall back to name availability + object-shaped arguments.

_OFFICE_TOOL_SCHEMAS: dict[str, dict[str, dict[str, tuple[type, ...]]]] = {
    "agent_peek": {"required": {}, "optional": {"target": (str,), "lines": (int,)}},
    "agent_send": {"required": {"name": (str,), "message": (str,)}, "optional": {}},
    "agent_status": {"required": {}, "optional": {"name": (str,)}},
    "bash": {"required": {"command": (str,)}, "optional": {"timeout": (int,)}},
    "coas_schedule_list": {"required": {}, "optional": {}},
    "edit": {"required": {"path": (str,), "edits": (list,)}, "optional": {}},
    "get_name": {"required": {}, "optional": {}},
    "goal_get": {"required": {}, "optional": {}},
    "kanban_snapshot": {"required": {}, "optional": {"detail": (str,), "task_id": (str,)}},
    "kill_agent": {"required": {"name": (str,)}, "optional": {"force": (bool,)}},
    "list_spawned": {"required": {}, "optional": {"name": (str,), "lines": (int,)}},
    "message_read": {"required": {}, "optional": {"limit": (int,)}},
    "message_send": {"required": {"channel": (str,), "message": (str,)}, "optional": {}},
    "read": {"required": {"path": (str,)}, "optional": {"offset": (int,), "limit": (int,)}},
    "rpc_send": {
        "required": {"name": (str,), "command": (str,), "message": (str,)},
        "optional": {"wait": (bool,), "timeout": (int,)},
    },
    "set_name": {"required": {"name": (str,)}, "optional": {}},
    "spawn_agent": {
        "required": {"name": (str,)},
        "optional": {"brief": (dict,), "cwd": (str,), "systemPrompt": (str,), "task": (str,)},
    },
    "write": {"required": {"path": (str,), "content": (str,)}, "optional": {}},
}

# task_type -> tools that must appear in the emitted calls for the record to
# count as a completed state transition (validator kind "state-transition").
# Execution-verified state checks (spec 0002 Stage 2) are out of scope here;
# this checks that the required mutating/handoff tools were invoked.
STATE_TOOLS_BY_TASK_TYPE: dict[str, tuple[str, ...]] = {
    "inventory-update": (
        "write", "edit", "bash", "message_send", "agent_send", "rpc_send", "spawn_agent",
    ),
    "agent-handoff": ("agent_send", "message_send", "spawn_agent", "rpc_send"),
}

# --- rendering ---------------------------------------------------------------

# The tool-call text format used across the corpus and the training data
# (Hermes/Qwen style), assembled from parts so this source file never needs
# the raw tag tokens in a string literal.
TOOL_CALL_OPEN = "<" + "tool" + "_call>"
TOOL_CALL_CLOSE = "<" + "/" + "tool" + "_call>"

TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)

_PLACEHOLDER_RE = re.compile(
    r"\{\{\s*[A-Za-z_][A-Za-z0-9_]*\s*\}\}|<TODO[^>]*>|\[TODO\]|<<[A-Z_]+>>"
)

# Markup that must never leak into a final, user-facing answer.
_LEAK_MARKERS: tuple[tuple[str, str], ...] = (
    (TOOL_CALL_OPEN, "tool-call markup"),
    ("trajectory truncated", "truncation marker"),
    ("[TRUNCATED]", "truncation marker"),
)

_CODE_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def render_system_prompt(record: dict[str, Any]) -> str:
    """Office-assistant system prompt for a spec-0001 record."""
    tools = ", ".join(record.get("context", {}).get("tools") or []) or "none"
    domain = record.get("domain") or "office"
    example = json.dumps({"name": "<tool>", "arguments": {}}, ensure_ascii=False)
    return (
        f"You are the {domain} assistant for the executive office on the DGX Spark. "
        "Work through the user's task step by step.\n"
        f"Available tools: {tools}.\n"
        "To call a tool, emit a block of the form\n"
        f"{TOOL_CALL_OPEN}\n{example}\n{TOOL_CALL_CLOSE}\n"
        "then wait for each tool result before continuing. Produce the required "
        "final output exactly when the task is complete."
    )


def render_user_prompt(record: dict[str, Any]) -> str:
    """User turn for a spec-0001 record."""
    return str(record.get("prompt") or "").strip()


def format_tool_call(call: dict[str, Any]) -> str:
    """Render a tool call in the canonical text format."""
    payload = json.dumps(
        {"name": call.get("name", ""), "arguments": call.get("args", call.get("arguments", {}))},
        ensure_ascii=False,
    )
    return TOOL_CALL_OPEN + "\n" + payload + "\n" + TOOL_CALL_CLOSE


def serialize_trajectory(record: dict[str, Any]) -> str:
    """Serialize the reference trajectory's assistant tool calls, in order,
    as canonical tagged blocks.

    Tool results are environment outputs, not model emissions, so they are
    deliberately NOT part of the serialized response.
    """
    blocks: list[str] = []
    for event in record.get("reference_trajectory") or []:
        if (
            isinstance(event, dict)
            and event.get("role") == "assistant"
            and isinstance(event.get("tool_call"), dict)
        ):
            blocks.append(format_tool_call(event["tool_call"]))
    return "\n\n".join(blocks)


def record_to_chat(record: dict[str, Any]) -> list[dict[str, str]]:
    """Chat-ready single-turn messages for one spec-0001 record.

    system = office-assistant prompt with the record's available tools;
    user   = the task prompt;
    assistant = the tool trajectory serialized into the response (ordered
    tool-call blocks) followed by the final output.
    """
    parts = [serialize_trajectory(record), str(record.get("final_output") or "").strip()]
    response = "\n\n".join(part for part in parts if part)
    return [
        {"role": "system", "content": render_system_prompt(record)},
        {"role": "user", "content": render_user_prompt(record)},
        {"role": "assistant", "content": response},
    ]


def generation_context(record: dict[str, Any]) -> list[dict[str, str]]:
    """Initial [system, user] messages for generating on a record."""
    return [
        {"role": "system", "content": render_system_prompt(record)},
        {"role": "user", "content": render_user_prompt(record)},
    ]


def replay_tool_results(record: dict[str, Any]) -> list[str]:
    """Ordered recorded tool results from the reference trajectory."""
    return [
        str(event.get("result") or "")
        for event in record.get("reference_trajectory") or []
        if isinstance(event, dict) and event.get("role") == "tool"
    ]



# --- JSONL I/O and record schema validation ---------------------------------

def iter_jsonl(path: Path | str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (line_number, record) pairs from a JSONL file.

    Raises ValueError (with line context) on unparseable lines so callers can
    fail loudly on a dirty corpus.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record is not a JSON object")
            yield line_number, record


def _is_str_nonempty(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def sha256_of_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_tool_calls(record: dict[str, Any]) -> int:
    """Number of assistant tool-call events in the reference trajectory."""
    return sum(
        1
        for event in record.get("reference_trajectory") or []
        if isinstance(event, dict)
        and event.get("role") == "assistant"
        and isinstance(event.get("tool_call"), dict)
    )


def classify_tier(record: dict[str, Any]) -> str:
    """Quality tier per the spec-0001 quality gate.

    gold = >= GOLD_MIN_TOOL_CALLS tool calls in the reference trajectory AND
    a final_output of at least GOLD_MIN_FINAL_OUTPUT_CHARS characters;
    silver = the rest.
    """
    if count_tool_calls(record) < GOLD_MIN_TOOL_CALLS:
        return "silver"
    final_output = record.get("final_output")
    if not isinstance(final_output, str) or len(final_output.strip()) < GOLD_MIN_FINAL_OUTPUT_CHARS:
        return "silver"
    return "gold"


def validate_record(record: dict[str, Any], label: str) -> list[str]:
    """Validate one corpus record against the spec-0001 section-5 schema.

    Returns a list of error strings (empty when valid). Lenient about unknown
    extra keys, strict about required fields, types, and enums.
    """
    errors: list[str] = []

    if not _is_str_nonempty(record.get("task_id")):
        errors.append(f"{label}: task_id must be a non-empty string")

    if record.get("domain") not in DOMAINS:
        errors.append(f"{label}: domain must be one of {list(DOMAINS)}, got {record.get('domain')!r}")

    if record.get("task_type") not in TASK_TYPES:
        errors.append(
            f"{label}: task_type must be one of {list(TASK_TYPES)}, got {record.get('task_type')!r}"
        )

    if record.get("difficulty") not in DIFFICULTIES:
        errors.append(
            f"{label}: difficulty must be one of {list(DIFFICULTIES)}, got {record.get('difficulty')!r}"
        )

    if record.get("source") not in SOURCES:
        errors.append(f"{label}: source must be one of {list(SOURCES)}, got {record.get('source')!r}")

    if not _is_str_nonempty(record.get("prompt")):
        errors.append(f"{label}: prompt must be a non-empty string")

    context = record.get("context")
    if not isinstance(context, dict):
        errors.append(f"{label}: context must be an object")
    else:
        tools = context.get("tools")
        if not isinstance(tools, list) or not all(_is_str_nonempty(t) for t in tools):
            errors.append(f"{label}: context.tools must be a list of non-empty strings")
        if not isinstance(context.get("prior_state"), dict):
            errors.append(f"{label}: context.prior_state must be an object")

    trajectory = record.get("reference_trajectory")
    if not isinstance(trajectory, list):
        errors.append(f"{label}: reference_trajectory must be a list")
    else:
        for index, event in enumerate(trajectory):
            where = f"{label}: trajectory[{index}]"
            if not isinstance(event, dict) or event.get("role") not in TRAJECTORY_ROLES:
                errors.append(f"{where}: must be an object with role 'assistant' or 'tool'")
                continue
            if event["role"] == "assistant":
                call = event.get("tool_call")
                if not isinstance(call, dict) or not _is_str_nonempty(call.get("name")):
                    errors.append(f"{where}: tool_call.name must be a non-empty string")
                if not isinstance(call, dict) or not isinstance(call.get("args"), dict):
                    errors.append(f"{where}: tool_call.args must be an object")
            elif not isinstance(event.get("result"), str):
                errors.append(f"{where}: tool result must be a string")

    if not _is_str_nonempty(record.get("final_output")):
        errors.append(f"{label}: final_output must be a non-empty string")

    validator = record.get("validator")
    if not isinstance(validator, dict) or validator.get("kind") not in VALIDATOR_KINDS:
        errors.append(
            f"{label}: validator.kind must be one of {list(VALIDATOR_KINDS)}, "
            f"got {(validator or {}).get('kind') if isinstance(validator, dict) else validator!r}"
        )
    elif not isinstance(validator.get("spec"), dict):
        errors.append(f"{label}: validator.spec must be an object")

    needs_validation = record.get("needs_validation")
    if needs_validation is not None and not isinstance(needs_validation, bool):
        errors.append(f"{label}: needs_validation must be a boolean when present")

    return errors


def validate_sft_record(record: dict[str, Any], label: str) -> list[str]:
    """Validate one emitted SFT dataset record (build_dataset.py output)."""
    errors: list[str] = []
    if not _is_str_nonempty(record.get("task_id")):
        errors.append(f"{label}: task_id must be a non-empty string")
    if record.get("tier") not in ("gold", "silver"):
        errors.append(f"{label}: tier must be 'gold' or 'silver', got {record.get('tier')!r}")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append(f"{label}: messages must be a list of at least 2 turns")
        return errors
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in (
            "system", "user", "assistant", "tool",
        ):
            errors.append(f"{label}: messages[{index}] must have a valid role")
        elif not isinstance(message.get("content"), str):
            errors.append(f"{label}: messages[{index}].content must be a string")
    if messages[0].get("role") != "system":
        errors.append(f"{label}: first message must be the system turn")
    if messages[1].get("role") != "user":
        errors.append(f"{label}: second message must be the user turn")
    if messages[-1].get("role") != "assistant":
        errors.append(f"{label}: last message must be the final assistant turn")
    return errors


# --- tool-call parsing and schema validation ---------------------------------

def _iter_bare_objects(text: str) -> Iterator[str]:
    """Yield balanced top-level {...} substrings, ignoring braces in strings."""
    start = -1
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start:index + 1]
                start = -1


def _normalize_call(payload: Any) -> dict[str, Any] | None:
    """Normalize a parsed payload into {"name": str, "args": Any} or None."""
    if not isinstance(payload, dict):
        return None
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name") or payload.get("name")
        args = function.get("arguments", payload.get("arguments", payload.get("parameters", {})))
    else:
        name = payload.get("name") or payload.get("tool")
        args = payload.get("arguments", payload.get("parameters", payload.get("args", {})))
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass  # keep raw string; validate_tool_call flags it
    return {"name": name.strip(), "args": args}


def parse_tool_calls(text: Any) -> list[dict[str, Any]]:
    """Extract tool calls from generated text.

    Parses canonical tagged blocks first, then falls back to bare JSON
    objects shaped like {"name": ..., "arguments"/"parameters": ...}.
    Returns a list of {"name": str, "args": Any} in order of appearance.
    """
    if not isinstance(text, str) or not text:
        return []
    raws = [match.group(1) for match in TOOL_CALL_BLOCK_RE.finditer(text)]
    if not raws:
        raws = [
            candidate
            for candidate in _iter_bare_objects(text)
            if '"name"' in candidate and ('"arguments"' in candidate or '"parameters"' in candidate)
        ]
    calls: list[dict[str, Any]] = []
    for raw in raws:
        try:
            payload = json.loads(raw)
        except ValueError:
            continue
        call = _normalize_call(payload)
        if call is not None:
            calls.append(call)
    return calls


def tool_calls_from_api_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract native OpenAI-style tool_calls from a chat completion message."""
    calls: list[dict[str, Any]] = []
    for entry in message.get("tool_calls") or []:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass  # keep raw string; validate_tool_call flags it
        if isinstance(name, str) and name.strip():
            calls.append({"name": name.strip(), "args": args})
    return calls


def merge_tool_calls(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate tool-call lists, dropping duplicates by (name, args)."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for call in group:
            key = (
                call.get("name", ""),
                json.dumps(call.get("args"), sort_keys=True, ensure_ascii=False),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(call)
    return merged


def validate_tool_call(name: Any, args: Any, available_tools: list[str] | None) -> list[str]:
    """Validate one emitted tool call against the office tool schemas.

    Checks name presence/type, availability in the record's tool list,
    object-shaped arguments, and the per-tool required/optional argument
    types. Returns a list of failure reasons (empty = valid).
    """
    reasons: list[str] = []
    if not isinstance(name, str) or not name.strip():
        return ["tool name missing or not a string"]
    name = name.strip()
    if available_tools is not None and name not in available_tools:
        reasons.append(f"tool {name!r} is not in the record's available tools")
    if not isinstance(args, dict):
        reasons.append(f"{name}: arguments must be a JSON object")
        return reasons
    schema = _OFFICE_TOOL_SCHEMAS.get(name)
    if schema is None:
        return reasons  # unknown tool: availability + object shape is all we can check
    for key, types in schema["required"].items():
        if key not in args:
            reasons.append(f"{name}: missing required argument {key!r}")
        elif not isinstance(args[key], types):
            reasons.append(f"{name}: argument {key!r} has wrong type (want {types})")
    for key, types in schema.get("optional", {}).items():
        if key in args and not isinstance(args[key], types):
            reasons.append(f"{name}: optional argument {key!r} has wrong type (want {types})")
    if name == "edit" and isinstance(args.get("edits"), list):
        for index, item in enumerate(args["edits"]):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("oldText"), str)
                or not isinstance(item.get("newText"), str)
            ):
                reasons.append(f"{name}: edits[{index}] must contain string oldText/newText")
    return reasons


# --- record validators (spec 0001 section 5 minimum set) ---------------------

def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    return match.group(1) if match else text


def format_class(text: Any) -> str:
    """Coarse byte-schema class of a final output: "json" or "prose"."""
    if not isinstance(text, str):
        return "empty"
    stripped = _strip_code_fence(text.strip())
    if not stripped:
        return "empty"
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return "json"
        except ValueError:
            pass
    return "prose"


def _check_clean_output(text: str, reasons: list[str]) -> None:
    if _PLACEHOLDER_RE.search(text):
        reasons.append("unfilled template placeholders present")
    for marker, label in _LEAK_MARKERS:
        if marker in text:
            reasons.append(f"{label} leaked into the output")


def check_template(record: dict[str, Any], output: Any) -> tuple[bool, list[str]]:
    """Template fidelity check (validator kind "template-match").

    Uses explicit template hints from validator.spec when present
    ("required_substrings", "template" as substring-or-regex), plus generic
    byte-schema rules: substantive length, no unfilled placeholders, no
    leaked tool/truncation markup, and a format class (JSON vs prose)
    matching the record's reference final_output.
    """
    spec = (record.get("validator") or {}).get("spec") or {}
    reasons: list[str] = []
    text = output.strip() if isinstance(output, str) else ""
    if len(text) < 30:
        reasons.append(f"final output too short ({len(text)} chars, want >= 30)")
    _check_clean_output(text, reasons)

    required = spec.get("required_substrings")
    if isinstance(required, list):
        for needle in required:
            if isinstance(needle, str) and needle not in text:
                reasons.append(f"missing required substring {needle!r}")

    template = spec.get("template")
    if isinstance(template, str) and template:
        try:
            if not re.search(template, text):
                reasons.append("output does not match the record template regex")
        except re.error:
            if template not in text:
                reasons.append("output does not contain the record template")

    reference = record.get("final_output")
    if isinstance(reference, str) and reference.strip():
        reference_class = format_class(reference)
        output_class = format_class(text)
        if reference_class != output_class:
            reasons.append(
                f"output format class {output_class!r} differs from reference {reference_class!r}"
            )
    return not reasons, reasons


def check_state_transition(
    record: dict[str, Any], call_results: list[dict[str, Any]], output: Any
) -> tuple[bool, list[str]]:
    """State-transition check (validator kind "state-transition").

    Verifies that the emitted calls include the state-changing tools the
    task type requires (validator.spec "expected_tools" overrides the
    built-in table). Execution-verified post-state comparison is future
    work (needs the office runtime; see spec 0002 Stage 2).
    """
    spec = (record.get("validator") or {}).get("spec") or {}
    reasons: list[str] = []
    expected = spec.get("expected_tools")
    if not isinstance(expected, list) or not expected:
        expected = list(STATE_TOOLS_BY_TASK_TYPE.get(record.get("task_type") or "", ()))
    called = {call.get("name") for call in call_results if isinstance(call, dict)}
    if expected and not (set(expected) & called):
        reasons.append(
            f"no state-transition tool called; expected one of {sorted(expected)}"
        )
    text = output.strip() if isinstance(output, str) else ""
    _check_clean_output(text, reasons)
    return not reasons, reasons


def _check_json_schema_subset(schema: dict[str, Any], value: Any, where: str) -> list[str]:
    """Tiny JSON-schema subset: type, required, properties (types only)."""
    errors: list[str] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        actual = (
            "object" if isinstance(value, dict)
            else "array" if isinstance(value, list)
            else "string" if isinstance(value, str)
            else "number" if isinstance(value, (int, float)) and not isinstance(value, bool)
            else "boolean" if isinstance(value, bool)
            else "null" if value is None
            else "unknown"
        )
        if actual != expected_type:
            errors.append(f"{where}: type {actual!r} != {expected_type!r}")
            return errors
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{where}: missing required key {key!r}")
        properties = schema.get("properties") or {}
        if isinstance(properties, dict):
            for key, sub in properties.items():
                if key in value and isinstance(sub, dict):
                    errors.extend(_check_json_schema_subset(sub, value[key], f"{where}.{key}"))
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(_check_json_schema_subset(schema["items"], item, f"{where}[{index}]"))
    return errors


def check_json_schema(record: dict[str, Any], output: Any) -> tuple[bool, list[str]]:
    """JSON-schema check (validator kind "json-schema").

    If validator.spec carries a "json_schema", validate a minimal subset of
    it (type / required / properties / items); otherwise the output must at
    least parse as JSON.
    """
    spec = (record.get("validator") or {}).get("spec") or {}
    reasons: list[str] = []
    text = output.strip() if isinstance(output, str) else ""
    parsed: Any = None
    if not text:
        reasons.append("empty output")
        return False, reasons
    try:
        parsed = json.loads(_strip_code_fence(text))
    except ValueError as exc:
        reasons.append(f"output does not parse as JSON ({exc})")
        return False, reasons
    schema = spec.get("json_schema")
    if isinstance(schema, dict):
        reasons.extend(_check_json_schema_subset(schema, parsed, "output"))
    _check_clean_output(text, reasons)
    return not reasons, reasons


def check_completion(
    record: dict[str, Any],
    output: Any,
    call_results: list[dict[str, Any]],
    validator_ok: bool | None,
    validator_reasons: list[str],
) -> tuple[bool, list[str]]:
    """End-to-end completion check for one record (spec 0001 section 6).

    A record is completed when: the model produced a substantive, clean
    final answer; it emitted at least one schema-valid tool call wherever
    the reference trajectory used tools (or none were needed); and the
    record's own validator check passed.
    """
    reasons: list[str] = []
    text = output.strip() if isinstance(output, str) else ""
    if len(text) < 20:
        reasons.append(f"no substantive final answer ({len(text)} chars)")
    _check_clean_output(text, reasons)
    if count_tool_calls(record) >= 1 and not call_results:
        reasons.append("reference trajectory used tools but the model emitted no tool calls")
    if call_results and not any(call.get("valid") for call in call_results):
        reasons.append("model emitted tool calls but none validated against the office tool schemas")
    if validator_ok is False:
        reasons.extend(validator_reasons or ["record validator check failed"])
    elif validator_ok is None:
        reasons.append("record validator could not be evaluated (unknown kind)")
    return not reasons, reasons
