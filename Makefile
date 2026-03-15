# Conformal Makefile
# Developer workflow for build, test, and packaging.

DOTNET = dotnet

# Default target
.PHONY: all
all: build test

# --- Local targets ---

.PHONY: build
build:
	$(DOTNET) build Conformal.sln

.PHONY: test
test:
	$(DOTNET) run --project src/analyzer/ConformalAnalyzer.fsproj -- --tests --quiet

.PHONY: test-verbose
test-verbose:
	$(DOTNET) run --project src/analyzer/ConformalAnalyzer.fsproj -- --tests

.PHONY: test-props
test-props:
	$(DOTNET) run --project src/analyzer/ConformalAnalyzer.fsproj -- --test-props

.PHONY: fable
fable:
	$(DOTNET) tool restore
	$(DOTNET) fable vscode-conformal/fable/ConformalFable.fsproj --outDir vscode-conformal/src/fable-out

.PHONY: extension
extension: fable
	cd vscode-conformal && node esbuild.mjs
	cd vscode-conformal && npx @vscode/vsce package --allow-missing-repository

.PHONY: analyze
analyze:
	@if [ -z "$(FILE)" ]; then echo "Usage: make analyze FILE=path/to/file.m"; exit 1; fi
	$(DOTNET) run --project src/analyzer/ConformalAnalyzer.fsproj -- $(FILE)

.PHONY: test-migrate
test-migrate:
	$(DOTNET) run --project src/migrate/ConformalMigrate.fsproj -- --test-migrate

.PHONY: verify-migrate
verify-migrate:
	$(DOTNET) build src/migrate/ConformalMigrate.fsproj -c Release
	python3 tools/verify_migrate.py

.PHONY: release
release:
	$(DOTNET) build src/analyzer/ConformalAnalyzer.fsproj -c Release

.PHONY: clean
clean:
	$(DOTNET) clean Conformal.sln
	rm -rf src/core/bin src/core/obj
	rm -rf src/shared/bin src/shared/obj
	rm -rf src/analyzer/bin src/analyzer/obj
	rm -rf src/migrate/bin src/migrate/obj

# --- Docker targets ---

DOCKER_IMAGE = conformal

.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE):latest --target runtime .
	docker build -t $(DOCKER_IMAGE):test --target test .

.PHONY: docker-test
docker-test:
	docker run --rm $(DOCKER_IMAGE):test --tests --quiet

.PHONY: docker-analyze
docker-analyze:
	@if [ -z "$(FILE)" ]; then echo "Usage: make docker-analyze FILE=path/to/file.m"; exit 1; fi
	docker run --rm -v "$$(pwd):/workspace" -w /workspace $(DOCKER_IMAGE):latest $(FILE)
