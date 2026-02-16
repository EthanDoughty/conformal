.PHONY: test run clean install install-vscode publish-vscode uninstall

PY=python3

test:
	python3 run_all_tests.py

run:
	@if [ -z "$(FILE)" ]; then echo "Usage: make run FILE=tests/basics/valid_add.m"; exit 1; fi
	$(PY) conformal.py "$(FILE)"

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
	find . -type f -name "*.pyc" -delete

install:
	pip install -e '.[lsp]'
	@echo "Conformal installed. Run 'conformal --tests' to verify."

install-vscode: install
	cd vscode-conformal && npm install && npm run compile && npx @vscode/vsce package
	@echo "Install .vsix: code --install-extension vscode-conformal/*.vsix"

publish-vscode: install-vscode
	cd vscode-conformal && npx @vscode/vsce publish

uninstall:
	pip uninstall -y conformal
