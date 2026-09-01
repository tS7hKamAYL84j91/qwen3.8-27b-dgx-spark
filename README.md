# Qwen3.8-27B on DGX Spark

This repository contains the small runtime I use to serve Qwen3.8-27B on a
single NVIDIA DGX Spark. The recommended profile uses RadixArk's NVFP4 target,
its native DSpark drafter, and SGLang. It exposes an OpenAI-compatible API on
localhost. The previous Unsloth NVFP4 + vLLM MTP launcher remains available as
a fallback and comparison profile.

The launcher is deliberately narrow in scope. It pins the container image,
keeps model weights outside the repository, and makes the settings that are
useful to tune available as environment variables.

## Requirements

- NVIDIA DGX Spark, or another GB10 system with `sm_121a` support
- Docker with NVIDIA Container Toolkit configured
- the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
  or `uvx` as a fallback

The default model directory is `$HOME/models/qwen3.8-27b`:

```text
qwen3.8-27b/
├── bf16/
├── fp8/
├── nvfp4/
├── radix-nvfp4/
└── radix-dspark/
```

Only the directory for the profile you run is required. If it is missing, the
launcher downloads the selected checkpoints at pinned revisions before
starting the server. Hugging Face's local-directory metadata makes interrupted
and repeated downloads resumable. Existing checkpoints are left alone.

## Running it

`make setup` ensures the promoted runtime is serving and is safe to re-run
(it starts the container only when the endpoint is not already healthy):

```bash
make setup
```

The recommended production command is:

```bash
runtime/run-sglang.sh
```

This selects the RadixArk NVFP4 target and DSpark drafter at block size 7,
an FP8 KV cache, FlashInfer attention, CUDA graphs for decode and verification,
Radix prefix caching, 8,192-token chunked prefill, and eight request slots.

The endpoint is available at `http://127.0.0.1:18083/v1`. For example:

```bash
curl http://127.0.0.1:18083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

Use `runtime/stop-sglang.sh` to stop the container cleanly.

## Runtime profiles

`runtime/run-sglang.sh` is the promoted profile. `runtime/run-vllm.sh` remains
available for the BF16, FP8, and Unsloth NVFP4 checkpoints; its second argument
is `none` or `mtp`.

The promoted profile has:

- a 131,072-token context window
- an FP8 KV cache and 909,193 measured cache-token capacity
- target-verification and DSpark-draft CUDA graphs
- Radix prefix caching and chunked prefill
- Qwen3 reasoning and Qwen3 Coder tool-call parsers
- at most eight concurrent requests

Explicit API sampling values are left to the client. Qwen recommends these
presets in the [official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices):

| Mode | Thinking | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repetition penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thinking | enabled | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Instruct | disabled | `0.7` | `0.8` | `20` | `0.0` | `1.5` | `1.0` |

Set `chat_template_kwargs.enable_thinking` to `true` or `false` in the request
body. Do not include prior reasoning content in subsequent messages.

SGLang's native Responses endpoint uses `reasoning.effort: "none"` rather than
`chat_template_kwargs` to select non-thinking mode. The parent DGX Spark
unified API performs this mapping automatically. Its non-thinking streaming
path also translates SGLang Chat Completions events to Responses events to
work around a serializer bug in the pinned image; text and function-call
streaming are covered by integration tests.

## Speed on DGX Spark

These are three-run measurements from one DGX Spark with a 131,072-token
context, temperature 0, and top-p 1. The 16-bit baseline is the official BF16
checkpoint; Qwen does not publish this checkpoint as FP16.

| Checkpoint | Speculation | Short decode | Coding decode | Long prefill |
| --- | ---: | ---: | ---: | ---: |
| Qwen BF16 | none | 4.41 tok/s | 4.40 tok/s | 1,206 tok/s |
| Qwen FP8 | none | 7.87 tok/s | 7.84 tok/s | 906 tok/s |
| Unsloth NVFP4 | none | 11.20 tok/s | 11.20 tok/s | 1,315 tok/s |
| Unsloth NVFP4 | MTP, width 2 | 23.91 tok/s | 21.44 tok/s | 1,248 tok/s |
| Unsloth NVFP4 | MTP, width 3 | 26.84 tok/s | 22.98 tok/s | — |
| RadixArk NVFP4 | SGLang DSpark, block 7 | 39.60 tok/s | 30.16 tok/s | — |

The first five rows use vLLM. The final promoted row uses SGLang and achieved
163.30 aggregate tok/s at concurrency 8. On a 55,670-token repository prompt,
prefix reuse reduced TTFT from 41.93 seconds to 0.37 seconds and the repeated
request completed in 7.25 seconds. A Qwen Coder tool-call smoke test passed.

Pinned checkpoint revisions used for the comparison:

- `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- `Qwen/Qwen3.8-27B-FP8` at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- `unsloth/Qwen3.8-27B-NVFP4` at `16b6615af3548b88e2d8e382457bc705b00479cf`
- `RadixArk/Qwen3.8-27B-NVFP4` at `52d1adc5f38aa5ebf099c29ed7025ba34cfbb854`
- `RadixArk/Qwen3.8-27B-DSpark` at `85ef153be924f17ce4bf62726954eeaa4a73e854`

### Ollama comparison

`benchmarks/bench.py` measures both runtimes with identical raw prompts and
greedy decoding (median of 3 runs, 2026-09-01, this machine):

| Test | SGLang DSpark (NVFP4) | Ollama 0.33.2 (GGUF) |
| --- | ---: | ---: |
| Short decode, 256 tokens | 22.25 tok/s | 21.61 tok/s |
| Coding decode, 512 tokens | 30.87 tok/s | 31.09 tok/s |
| Long prefill, 37,388 tokens | 24.6 s (1,518 tok/s) | 53.6 s (697 tok/s) |

Decode is a tie because both runtimes use speculative decoding (DSpark
versus ollama's MTP). SGLang prefills 2.2x faster and serves 8 concurrent
requests against ollama's single slot, so it remains the promoted profile.
The two runtimes cannot hold the GPU at the same time on this box, which is
recorded in [ADR-0001](docs/adrs/0001-sglang-resident-runtime.md).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `QWEN38_MODELS_ROOT` | `$HOME/models/qwen3.8-27b` | checkpoint directories |
| `QWEN38_SGLANG_IMAGE` | pinned digest | SGLang container image |
| `QWEN38_CACHE` | `${XDG_CACHE_HOME:-$HOME/.cache}/qwen3.8-sglang` | compiler and kernel cache |
| `QWEN38_PORT` | `18083` | local API port |
| `QWEN38_CONTEXT` | `131072` | maximum model length |
| `QWEN38_MEMORY_FRACTION` | `0.80` | SGLang static memory fraction |
| `QWEN38_MAX_REQUESTS` | `8` | maximum concurrent requests |
| `QWEN38_CHUNKED_PREFILL_SIZE` | `8192` | prefill chunk size |
| `QWEN38_KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache data type |
| `QWEN38_DSPARK_BLOCK_SIZE` | `7` | DSpark proposal block size |
| `QWEN38_AUTO_DOWNLOAD` | `1` | set to `0` to require preloaded weights |

The pinned image digest is the only one tested here. The launcher uses host
networking and binds SGLang to `127.0.0.1`; put an authenticated proxy in front
of it if remote access is needed. The public checkpoints do not require
authentication; if that changes, the downloader honors the Hugging Face CLI's
saved login or `HF_TOKEN`.

## Notes

Model licenses are not covered by this repository's license. Review the terms
for each checkpoint before downloading or redistributing weights. The launcher
also enables vLLM's `--trust-remote-code`; inspect checkpoint code when changing
away from the pinned revisions above.
