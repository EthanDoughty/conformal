.PHONY: test run compare clean lm-up lm-down ai-local lm-status

PY=python3

test:
	python3 run_all_tests.py

run:
	@if [ -z "$(FILE)" ]; then echo "Usage: make run FILE=tests/test1.m"; exit 1; fi
	$(PY) mmshape.py "$(FILE)"

compare:
	@if [ -z "$(FILE)" ]; then echo "Usage: make compare FILE=tests/test1.m"; exit 1; fi
	$(PY) mmshape.py --compare "$(FILE)"

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
	find . -type f -name "*.pyc" -delete

# --- Local LM Studio Integration ---

LM_ID ?= local-qwen3-14b
LM_MODEL ?= qwen/qwen3-14b
LM_PORT ?= 1234
LM_CTX ?= 8192
LM_GPU ?= max
LM_TTL ?= 900

lm-up:
	@echo "Loading $(LM_MODEL) as $(LM_ID) (ctx=$(LM_CTX), gpu=$(LM_GPU))..."
	lms load "$(LM_MODEL)" --identifier "$(LM_ID)" --gpu "$(LM_GPU)" --context-length "$(LM_CTX)" --ttl "$(LM_TTL)" -y
	@echo "Starting server on port $(LM_PORT)..."
	lms server start -p "$(LM_PORT)"
	@echo "LM Studio ready at http://localhost:$(LM_PORT)/v1 (model id: $(LM_ID))"

lm-down:
	@echo "Stopping LM Studio server..."
	@lms server stop >/dev/null 2>&1 || true
	@echo "Unloading all models..."
	@lms unload --all >/dev/null 2>&1 || true
	@echo "LM Studio stopped."

ai-local:
	@if [ -z "$(ROLE)" ]; then echo "Usage: make ai-local ROLE=mentor-reviewer PROMPT='...'" ; exit 1; fi
	@if [ -z "$(PROMPT)" ]; then echo "Usage: make ai-local ROLE=mentor-reviewer PROMPT='...'" ; exit 1; fi
	@echo "$(PROMPT)" | python3 tools/ai_local.py "$(ROLE)"

lm-status:
	@echo "Loaded models:"
	@lms ps || true
	@echo ""
	@echo "Server check:"
	@curl -sS http://localhost:$(LM_PORT)/v1/models | head -c 300; echo