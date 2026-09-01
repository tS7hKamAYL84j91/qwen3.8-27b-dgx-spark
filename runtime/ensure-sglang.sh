#!/usr/bin/env bash
# Idempotently ensure the promoted SGLang runtime is serving.
# Safe to re-run: exits immediately when the endpoint is already healthy.
set -euo pipefail

port=${QWEN38_PORT:-18083}
container_name=${QWEN38_CONTAINER_NAME:-qwen38-sglang}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
log_file=${QWEN38_ENSURE_LOG:-${XDG_CACHE_HOME:-${HOME}/.cache}/qwen3.8-sglang/ensure.log}

healthy() {
  curl -fsS --max-time 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

if healthy; then
  echo "sglang already serving on 127.0.0.1:${port}"
  exit 0
fi

# A stopped-but-unremoved container would conflict with a fresh start.
if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "container $container_name exists but is not healthy; stopping it first" >&2
  bash "$script_dir/stop-sglang.sh" || true
fi

mkdir -p "$(dirname "$log_file")"
echo "starting sglang (container $container_name, port $port); logs: $log_file"
nohup bash "$script_dir/run-sglang.sh" >>"$log_file" 2>&1 &

# Allow up to 30 minutes: first start may download checkpoints (~20 GB).
for _ in $(seq 1 180); do
  if healthy; then
    echo "sglang is serving on 127.0.0.1:${port}"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$container_name"; then
    echo "container $container_name exited before becoming healthy; last log lines:" >&2
    tail -n 20 "$log_file" >&2 || true
    exit 1
  fi
  sleep 10
done

echo "sglang did not become healthy within 1800s; last log lines:" >&2
tail -n 20 "$log_file" >&2 || true
exit 1
