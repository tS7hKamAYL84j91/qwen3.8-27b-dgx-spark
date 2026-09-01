#!/usr/bin/env bash
# Sweep SGLang server configurations for decode throughput.
# Usage: sweep.sh <tag> <block_size> [extra server args...]
# Restarts the container with the given config, waits for health, benches
# short+coding decode (2 runs), writes JSON to /tmp/dspark_sweep/.
set -u

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <tag> <block_size> [extra server args...]" >&2
  exit 2
fi

tag=$1
bs=$2
shift 2
extra_args=$*
results_dir=/tmp/dspark_sweep
mkdir -p "$results_dir"

wait_healthy() {
  local deadline=$(( $(date +%s) + 300 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 3 "http://127.0.0.1:${QWEN38_PORT:-18083}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${QWEN38_CONTAINER_NAME:-qwen38-sglang}"; then
      echo "container exited during startup; last log lines:"
      tail -n 15 "$results_dir/serve_$tag.log" 2>/dev/null || true
      return 1
    fi
    sleep 5
  done
  echo "timed out waiting for health"
  return 1
}

echo "=== config $tag: block_size=$bs extra=[${extra_args:-none}] ==="
docker stop --timeout 20 "${QWEN38_CONTAINER_NAME:-qwen38-sglang}" >/dev/null 2>&1 || true
# Give the previous container time to fully release GPU memory; a too-early
# start dies silently at CUDA init (NVRM NO_MEMORY during teardown).
sleep 10

# shellcheck disable=SC2086
QWEN38_DSPARK_BLOCK_SIZE="$bs" QWEN38_EXTRA_ARGS="$extra_args" \
  nohup bash "$repo/runtime/run-sglang.sh" >"$results_dir/serve_$tag.log" 2>&1 &

if ! wait_healthy; then
  echo "config $tag attempt 1 failed; retrying after 10s drain"
  sleep 10
  # shellcheck disable=SC2086
  QWEN38_DSPARK_BLOCK_SIZE="$bs" QWEN38_EXTRA_ARGS="$extra_args" \
    nohup bash "$repo/runtime/run-sglang.sh" >"$results_dir/serve_$tag.log" 2>&1 &
  if ! wait_healthy; then
    echo "CONFIG $tag: UNHEALTHY"
    exit 1
  fi
fi

python3 "$repo/benchmarks/bench.py" --backend sglang --tests short,coding --runs 2 \
  --out "$results_dir/bench_$tag.json" 2>&1 | tail -4