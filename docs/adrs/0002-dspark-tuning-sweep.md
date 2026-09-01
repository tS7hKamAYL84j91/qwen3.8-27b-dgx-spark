# ADR-0002: Keep shipped SGLang defaults after a tuning sweep

- Status: Accepted
- Date: 2026-09-01
- Supersedes: nothing (complements ADR-0001)

## Context

After ADR-0001 made SGLang the resident runtime, a sweep searched for
single-stream decode throughput improvements over the promoted profile
(DSpark block size 7, unquantized drafter, 8,192-token chunked prefill,
FlashInfer attention). Each candidate was restarted cold and measured with
`benchmarks/bench.py` (greedy, identical prompts, 2-3 runs):

| Config | Short decode | Coding decode | Long prefill (37k) |
| --- | ---: | ---: | ---: |
| block 7 (baseline) | 22.25 tok/s | 30.87 tok/s | 1,518 tok/s |
| block 5 | 21.24 | 31.17 | — |
| block 9 | 19.77 | 29.92 | — |
| block 11 | 19.72 | 30.09 | — |
| block 7 + `nvfp4_online` drafter | 21.24 | 29.52 | — |
| block 7 + 16k prefill chunks | 21.33 | 29.70 | 921 tok/s |

## Decision

Keep the shipped defaults (block size 7, unquant drafter, 8,192-token chunks).
Recorded tooling: `QWEN38_EXTRA_ARGS` env passthrough in `runtime/run-sglang.sh`
and `benchmarks/sweep.sh` for reproducible config sweeps.

## Findings and constraints

- Decode sits at the GB10 memory-bandwidth wall for a 27B model; an
  independent ollama MTP run reached the same ~21-31 tok/s, confirming the
  hardware bound rather than a runtime deficiency.
- `--speculative-align-verify-tokens-to-graph-tier` appears in `--help` but is
  rejected by the image's argparse at runtime; treated as unavailable.
- The DSPARK drafter repo ships no `sps-table` / `confidence` artifacts;
  generating them is future work, not a configuration change.
- Lowering `--speculative-accept-threshold-single/acc` trades output quality
  for speed; not promoted without an explicit quality sign-off.

## Consequences

- `make setup` / `runtime/ensure-sglang.sh` remain the production path; no
  launcher defaults changed in this sweep beyond the additive
  `QWEN38_EXTRA_ARGS` passthrough.
- The sweep harness drains 10 s and retries once after a config restart; an
  early start can die silently at CUDA init during container teardown, and a
  transient `docker ps` failure must not be read as container death
  (three-consecutive-miss rule in `runtime/ensure-sglang.sh`).
