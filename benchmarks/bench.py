#!/usr/bin/env python3
"""Benchmark Qwen3.8-27B runtimes on DGX Spark: SGLang DSpark vs Ollama.

Measures TTFT, decode tok/s, and long-prefill throughput using identical raw
prompts and greedy decoding (temperature 0) so chat-template differences do
not skew the comparison.

- SGLang: OpenAI-compatible /v1/completions with stream_options.include_usage.
- Ollama: native /api/generate with raw=true (bypasses the GGUF chat template).

Each test is run `--runs` times after one warmup request; medians are reported.
Raw per-run JSON is written to --out for auditability.

Usage:
  python3 bench.py --backend sglang --tests short,coding,prefill \
      --runs 3 --out /tmp/sglang.json
  python3 bench.py --backend ollama --tests short,coding,prefill \
      --runs 3 --out /tmp/ollama.json
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
import uuid

SGLANG_BASE = "http://127.0.0.1:18083"
OLLAMA_BASE = "http://127.0.0.1:11434"

SHORT_PROMPT = (
    "Write a short explanation of how speculative decoding works, "
    "covering draft models, verification, and acceptance rates.\n\n"
)
CODING_PROMPT = (
    "#!/usr/bin/env python3\n"
    "import argparse, json, sys\n\n\n"
    "def stream_tokens(prompt, max_tokens, temperature):\n"
    '    """Stream completions from an OpenAI-compatible endpoint."""\n'
    "    payload = {\n"
    '        "model": model,\n'
    '        "prompt": prompt,\n'
    '        "max_tokens": max_tokens,\n'
    '        "temperature": temperature,\n'
    '        "stream": True,\n'
    "    }\n"
    "    with httpx.stream(\n"
    '        "POST", url, json=payload, timeout=600\n'
    "    ) as response:\n"
    "        for line in response.iter_lines():\n"
)

GEN_SHORT = 256
GEN_CODING = 512
GEN_PREFILL = 32
PREFILL_TARGET_TOKENS = 48000

# Deterministic filler used to build a ~55k-token prefill prompt.
FILLER_BLOCK = (
    "The scheduler maintains a run queue per device and drains it in rounds. "
    "Each round begins by scoring resident requests on age, priority, and "
    "remaining budget, then selects the highest scoring batch that still fits "
    "the memory ceiling. Requests which miss the cut are deferred to the next "
    "round and their age score grows so that starvation cannot persist. "
    "Prefix caching keys are computed from the tokenized prompt in blocks of "
    "sixteen tokens, and a hit allows the kernel to skip recomputation for "
    "that span entirely. Speculative verification runs the draft tokens "
    "through the target model in one batched forward pass and accepts the "
    "longest prefix that matches, falling back to one token when nothing "
    "matches. The KV cache is paged so that sequences can be evicted and "
    "restored without copying contiguous buffers.\n\n"
)


def http_post_json(url, payload, timeout=1800):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def build_long_prompt(target_tokens):
    # Estimate tokens at ~4 chars/token from a measured block, oversample 20%.
    try:
        n_blocks = max(1, int(target_tokens / (len(FILLER_BLOCK) / 4)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid prefill target token count: {target_tokens!r}"
        ) from exc
    parts = []
    for i in range(n_blocks):
        parts.append(f"[block {i:05d}] {FILLER_BLOCK}")
    return "".join(parts)


def bench_sglang(prompt, max_tokens, base):
    """Stream from /v1/completions; returns metrics dict or raises."""
    payload = {
        "model": "qwen3.8",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first_t = None
    last_t = None
    usage = None
    resp = http_post_json(f"{base}/v1/completions", payload)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"sglang: malformed SSE data payload: {body[:120]!r}"
            ) from exc
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if choices and (choices[0].get("text") or "") != "":
            now = time.perf_counter()
            if first_t is None:
                first_t = now
            last_t = now
    t_end = time.perf_counter()
    if usage is None:
        raise RuntimeError("sglang: no usage chunk received; include_usage unsupported?")
    completion_tokens = usage["completion_tokens"]
    prompt_tokens = usage["prompt_tokens"]
    if prompt_tokens >= 131072:
        raise RuntimeError(
            f"sglang: prompt truncated or over context ({prompt_tokens} >= 131072)"
        )
    ttft = (first_t - t0) if first_t else None
    decode_s = (last_t - first_t) if (first_t and last_t and last_t > first_t) else None
    decode_tps = (completion_tokens - 1) / decode_s if decode_s else None
    prefill_tps = prompt_tokens / ttft if ttft else None
    total_tps = completion_tokens / (t_end - t0) if t_end > t0 else None
    return {
        "backend": "sglang",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "total_tps": total_tps,
        "e2e_s": t_end - t0,
    }


def bench_ollama(prompt, max_tokens, base, num_ctx):
    """Stream from native /api/generate with raw=true."""
    payload = {
        "model": "qwen3.8:27b",
        "prompt": prompt,
        "raw": True,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
            "num_ctx": num_ctx,
        },
    }
    t0 = time.perf_counter()
    first_t = None
    last_t = None
    final = None
    resp = http_post_json(f"{base}/api/generate", payload)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"ollama: malformed JSONL frame: {line[:120]!r}"
            ) from exc
        if chunk.get("done"):
            final = chunk
            break
        if chunk.get("response"):
            now = time.perf_counter()
            if first_t is None:
                first_t = now
            last_t = now
    t_end = time.perf_counter()
    if final is None:
        raise RuntimeError("ollama: stream ended without done frame")
    completion_tokens = final.get("eval_count", 0)
    prompt_tokens = final.get("prompt_eval_count", 0)
    if prompt_tokens >= num_ctx:
        raise RuntimeError(
            f"ollama: prompt at/over num_ctx ({prompt_tokens} >= {num_ctx}); "
            "the server likely truncated it - reduce the prefill target"
        )
    ttft = (first_t - t0) if first_t else None
    decode_s = (last_t - first_t) if (first_t and last_t and last_t > first_t) else None
    decode_tps = (completion_tokens - 1) / decode_s if decode_s else None
    prefill_tps = prompt_tokens / ttft if ttft else None
    total_tps = completion_tokens / (t_end - t0) if t_end > t0 else None
    backend_tps = (
        completion_tokens / final["eval_duration"] * 1e9
        if final.get("eval_duration")
        else None
    )
    return {
        "backend": "ollama",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "total_tps": total_tps,
        "backend_eval_tps": backend_tps,
        "load_s": final.get("load_duration", 0) / 1e9,
        "e2e_s": t_end - t0,
    }


TESTS = {
    "short": {"gen": GEN_SHORT, "kind": "short"},
    "coding": {"gen": GEN_CODING, "kind": "coding"},
    "prefill": {"gen": GEN_PREFILL, "kind": "prefill"},
}


def variant_prompt(base_prompt, tag):
    """Prepend a unique header so prefix caches (SGLang radix, llama.cpp)
    cannot collapse repeated prefill runs into a cache hit."""
    return f"[bench variant {tag}\nignore this header and answer the text below]\n\n{base_prompt}"


def run_backend(backend, tests, runs, out_path):
    base = SGLANG_BASE if backend == "sglang" else OLLAMA_BASE
    long_prompt = build_long_prompt(PREFILL_TARGET_TOKENS) if "prefill" in tests else None
    results = {}
    for test in tests:
        cfg = TESTS[test]
        base_prompt = long_prompt if test == "prefill" else (
            CODING_PROMPT if test == "coding" else SHORT_PROMPT
        )
        num_ctx = 65536 if test == "prefill" else 8192
        # warmup (one throwaway request); unique prefix per request for prefill
        prompt = (
            variant_prompt(base_prompt, f"warmup-{uuid.uuid4().hex[:8]}")
            if test == "prefill"
            else base_prompt
        )
        print(f"[{backend}/{test}] warmup...", flush=True)
        if backend == "sglang":
            bench_sglang(prompt, cfg["gen"], base)
        else:
            bench_ollama(prompt, cfg["gen"], base, num_ctx)
        run_results = []
        for i in range(runs):
            print(f"[{backend}/{test}] run {i + 1}/{runs}...", flush=True)
            prompt = variant_prompt(base_prompt, f"run{i + 1}-{uuid.uuid4().hex[:8]}") if test == "prefill" else base_prompt
            if backend == "sglang":
                r = bench_sglang(prompt, cfg["gen"], base)
            else:
                r = bench_ollama(prompt, cfg["gen"], base, num_ctx)
            r["test"] = test
            r["run"] = i + 1
            run_results.append(r)
            print(
                f"  ttft={r['ttft_s']:.2f}s decode={r['decode_tps']:.2f} tok/s "
                f"out={r['completion_tokens']}tok prompt={r['prompt_tokens']}tok",
                flush=True,
            )
        results[test] = run_results
    try:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    except OSError as exc:
        raise RuntimeError(f"failed to write results to {out_path}: {exc}") from exc
    print(f"\nwrote {out_path}")
    return results


def summarize(results, runs):
    print(f"\n{'test':<10} {'ttft (s)':>12} {'decode tok/s':>14} {'prompt tok':>12} {'out tok':>9}")
    for test, rr in results.items():
        ttfts = [r["ttft_s"] for r in rr if r["ttft_s"]]
        decodes = [r["decode_tps"] for r in rr if r["decode_tps"]]
        ptoks = [r["prompt_tokens"] for r in rr]
        otoks = [r["completion_tokens"] for r in rr]
        fmt = lambda xs: f"{statistics.median(xs):.2f}" if xs else "-"
        print(
            f"{test:<10} {fmt(ttfts):>12} {fmt(decodes):>14} "
            f"{statistics.median(ptoks):>12.0f} {statistics.median(otoks):>9.0f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sglang", "ollama"], required=True)
    ap.add_argument("--tests", default="short,coding,prefill")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    results = run_backend(args.backend, tests, args.runs, args.out)
    summarize(results, args.runs)


if __name__ == "__main__":
    sys.exit(main())