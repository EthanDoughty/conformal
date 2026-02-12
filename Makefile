.PHONY: test run compare clean lm-up lm-down ai-local lm-status ask chat

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

# --- Local Ollama Integration ---

LM_MODEL ?= qwen2.5-coder:14b
LM_PORT ?= 11434
LM_CTX ?= 12288

lm-up:
	@echo "Pulling $(LM_MODEL)..."
	ollama pull "$(LM_MODEL)"
	@echo "Ollama ready at http://localhost:$(LM_PORT) (model: $(LM_MODEL))"

lm-down:
	@echo "Ollama manages model lifecycle automatically. To stop a running model:"
	@echo "  ollama stop \"$(LM_MODEL)\""

lm-status:
	@echo "Loaded models:"
	@ollama list || true
	@echo ""
	@echo "Server check:"
	@curl -sS http://localhost:$(LM_PORT)/api/tags | head -c 300; echo

ai-local:
	@if [ -z "$(Q)" ] && [ -z "$(PROMPT)" ]; then echo "Usage: make ai-local Q='...' [ROLE=mentor] [FILE=path]"; exit 1; fi
	$(PY) tools/ai_local.py --role "$(or $(ROLE),mentor)" $(if $(FILE),--file "$(FILE)") $(or $(Q),$(PROMPT))

ask:
	@if [ -z "$(Q)" ]; then echo "Usage: make ask Q='your question' [ROLE=mentor] [FILE=path]"; exit 1; fi
	$(PY) tools/ai_local.py --role "$(or $(ROLE),mentor)" $(if $(FILE),--file "$(FILE)") $(Q)

chat:
	$(PY) tools/ai_local.py --chat --role "$(or $(ROLE),mentor)" $(if $(FILE),--file "$(FILE)")
