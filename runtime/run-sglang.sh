#!/usr/bin/env bash
set -euo pipefail

container_image=${QWEN38_SGLANG_IMAGE:-lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1}
models_root=${QWEN38_MODELS_ROOT:-${HOME}/models/qwen3.8-27b}
cache_root=${XDG_CACHE_HOME:-${HOME}/.cache}
cache_dir=${QWEN38_CACHE:-${cache_root}/qwen3.8-sglang}
port=${QWEN38_PORT:-18083}
context=${QWEN38_CONTEXT:-131072}
mem_fraction=${QWEN38_MEMORY_FRACTION:-0.80}
max_requests=${QWEN38_MAX_REQUESTS:-8}
chunked_prefill_size=${QWEN38_CHUNKED_PREFILL_SIZE:-8192}
kv_cache_dtype=${QWEN38_KV_CACHE_DTYPE:-fp8_e4m3}
dspark_block_size=${QWEN38_DSPARK_BLOCK_SIZE:-7}
container_name=${QWEN38_CONTAINER_NAME:-qwen38-sglang}
auto_download=${QWEN38_AUTO_DOWNLOAD:-1}
# Intentional word splitting: QWEN38_EXTRA_ARGS carries multiple server flags.
# shellcheck disable=SC2086
extra_args=${QWEN38_EXTRA_ARGS:-}

target_repo=RadixArk/Qwen3.8-27B-NVFP4
target_revision=52d1adc5f38aa5ebf099c29ed7025ba34cfbb854
target_dir=$models_root/radix-nvfp4
draft_repo=RadixArk/Qwen3.8-27B-DSpark
draft_revision=85ef153be924f17ce4bf62726954eeaa4a73e854
draft_dir=$models_root/radix-dspark

download_checkpoint() {
  local repo=$1
  local revision=$2
  local destination=$3
  local marker=$destination/.qwen38-download-in-progress

  if [[ -s "$destination/config.json" && ! -e "$marker" ]]; then
    return 0
  fi
  if [[ "$auto_download" != 1 ]]; then
    echo "Qwen3.8 checkpoint is missing or incomplete: $destination" >&2
    echo "Automatic downloads are disabled by QWEN38_AUTO_DOWNLOAD=0." >&2
    exit 1
  fi

  mkdir -p "$destination"
  touch "$marker"
  echo "Downloading $repo at $revision to $destination..."
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --revision "$revision" --local-dir "$destination"
  elif command -v uvx >/dev/null 2>&1; then
    uvx --from huggingface-hub hf download \
      "$repo" --revision "$revision" --local-dir "$destination"
  elif [[ -x "$HOME/.local/bin/uvx" ]]; then
    "$HOME/.local/bin/uvx" --from huggingface-hub hf download \
      "$repo" --revision "$revision" --local-dir "$destination"
  else
    echo "Install the Hugging Face CLI or uv before downloading checkpoints:" >&2
    echo "  https://huggingface.co/docs/huggingface_hub/guides/cli" >&2
    exit 1
  fi

  if [[ ! -s "$destination/config.json" ]]; then
    echo "checkpoint download did not produce $destination/config.json" >&2
    exit 1
  fi
  rm -f "$marker"
}

download_checkpoint "$target_repo" "$target_revision" "$target_dir"
download_checkpoint "$draft_repo" "$draft_revision" "$draft_dir"
mkdir -p "$cache_dir"

exec docker run --rm \
  --name "$container_name" \
  --gpus all \
  --ipc host \
  --network host \
  -e TVM_FFI_GPU_BACKEND=cuda \
  -v "$target_dir:/model:ro" \
  -v "$draft_dir:/draft:ro" \
  -v "$cache_dir:/root/.cache" \
  "$container_image" \
  python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /model \
  --served-model-name qwen3.8 \
  --context-length "$context" \
  --kv-cache-dtype "$kv_cache_dtype" \
  --mem-fraction-static "$mem_fraction" \
  --max-running-requests "$max_requests" \
  --chunked-prefill-size "$chunked_prefill_size" \
  --tp-size 1 \
  --mamba-full-memory-ratio 3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /draft \
  --speculative-dspark-block-size "$dspark_block_size" \
  --speculative-draft-model-quantization unquant \
  --attention-backend flashinfer \
  --disable-prefill-cuda-graph \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 127.0.0.1 \
  --port "$port" \
  $extra_args
