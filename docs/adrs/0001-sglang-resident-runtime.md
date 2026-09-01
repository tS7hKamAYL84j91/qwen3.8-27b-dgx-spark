# ADR-0001: SGLang is the resident Qwen3.8 runtime; ollama hosts small models only

- Status: Accepted
- Date: 2026-09-01
- Decided by: Gravitas (direction relayed in the qwen3.8-27b-dgx-spark GM session); ratification relayed via chief-of-staff

## Context

The DGX Spark (GB10) exposes a single 121.7 GiB unified CPU/GPU memory pool.
The promoted SGLang DSpark runtime reserves roughly 91 GiB, leaving about
17-20 GiB for everything else. Ollama 0.33.2 (upgraded 2026-09-01) now starts
models at their full native context by default (262,144 tokens for
qwen3.8:27b), which pre-allocates a 17 GiB KV cache on top of the weights.

Consequences measured on 2026-09-01:

- ollama qwen3.8:27b needs ~33 GiB and failed to load seven consecutive times
  (`cudaMalloc failed: out of memory ... failed to allocate buffer for kv
  cache`), returning 503s to callers.
- Every failed load cycle stressed the host: 121/121 GiB RAM used, 10 GiB
  swap, kernel OOM kills, and NVRM driver OOM errors.
- gemma4:26b coexists fine (Gemma sliding-window attention keeps its 262k KV
  cache at ~1 GiB), while qwen3.6:latest (23 GiB weights) cannot fit at all.

An A/B benchmark (greedy decode, identical raw prompts, 3 runs) found decode
throughput effectively tied (~22 tok/s short, ~31 tok/s coding) because both
runtimes use speculative decoding, while SGLang prefill was 2.2x faster
(1,518 vs 697 tok/s on a 37,388-token prompt) and SGLang serves 8 concurrent
requests versus ollama's 1. Raw data: /tmp/bench_sglang_sc.json,
/tmp/bench_sglang_pf2.json, /tmp/bench_ollama_sc.json, /tmp/bench_ollama_pf.json
(reproducible with `benchmarks/bench.py`).

## Decision

1. SGLang is the resident runtime for Qwen3.8-27B on this machine. `make
   setup` (via `runtime/ensure-sglang.sh`) idempotently ensures it is serving.
2. Remove the ollama models that cannot coexist with the resident runtime:
   `qwen3.8:27b` and `qwen3.6:latest` (~39 GiB freed). Re-downloading either
   is a single `ollama pull` if ever needed.
3. Keep the ollama models that fit alongside SGLang: gemma4:26b (proven),
   gpt-oss:20b, lfm2.5. gemma4:31b (19 GiB) is borderline: it has not failed
   in evidence so far but has not been exercised under residency either.

## Alternatives considered

- Cap ollama's default context (OLLAMA_CONTEXT_LENGTH=16384) to make large
  models fit: rejected because qwen3.6:latest's weights alone exceed the free
  memory, and a shared-infra env change would degrade every other local model.
- Keep both large ollama models for manual benchmarking: rejected because
  failed loads repeatedly degraded the whole host (swap thrash, OOM kills).

## Consequences

- Benchmarking ollama models that need >17 GiB requires stopping SGLang first
  (make stop, benchmark, make setup).
- Coas routing that expects ollama/qwen3.6:latest falls back to cloud members;
  gemma4:31b should be validated under residency or the Navigator slot moved
  to gemma4:26b.
- This ADR supersedes no earlier record (ADR-0001 is the first); the SGLang
  promotion itself remains documented in the README.
