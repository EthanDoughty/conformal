.PHONY: test run compare clean

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