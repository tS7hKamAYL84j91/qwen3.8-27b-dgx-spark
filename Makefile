.PHONY: setup stop status

# Ensure the promoted SGLang runtime is serving; safe to re-run.
# Waits for first-start checkpoint downloads (up to 30 minutes).
setup:
	bash runtime/ensure-sglang.sh

# Stop the SGLang container cleanly.
stop:
	bash runtime/stop-sglang.sh

# Report whether the endpoint is healthy.
status:
	@curl -fsS --max-time 3 "http://127.0.0.1:$${QWEN38_PORT:-18083}/v1/models" >/dev/null 2>&1 && echo "sglang: up on 127.0.0.1:$${QWEN38_PORT:-18083}" || echo "sglang: down"