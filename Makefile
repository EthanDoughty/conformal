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
	cd src && \
	  $(DOTNET) tool restore && \
	  $(DOTNET) fable ../vscode-conformal/fable/ConformalFable.fsproj --outDir ../vscode-conformal/src/fable-out

.PHONY: extension
extension: fable
	cd vscode-conformal && node esbuild.mjs
	cd vscode-conformal && npx @vscode/vsce package --allow-missing-repository

# Publishes the packaged VSIX to the Marketplace. The PAT is an Azure DevOps
# token (scope Marketplace:Manage, all orgs) read from $VSCE_PAT, else from a
# gitignored file under ~/.config. PATs expire, which is the usual reason a
# release fails to ship, so 'make verify-pat' checks it before a release.
PAT_FILE ?= $(HOME)/.config/conformal/vsce-pat

.PHONY: publish
publish: extension
	cd vscode-conformal && \
	  VER=$$(node -p "require('./package.json').version") && \
	  PAT="$${VSCE_PAT:-$$(cat $(PAT_FILE) 2>/dev/null)}" && \
	  { [ -n "$$PAT" ] || { echo "No PAT found. Put it in $(PAT_FILE) or export VSCE_PAT." >&2; exit 1; }; } && \
	  VSCE_PAT="$$PAT" npx @vscode/vsce publish --packagePath conformal-$$VER.vsix

# Checks that the stored PAT is still valid, so expiry is caught before a release.
.PHONY: verify-pat
verify-pat:
	@cd vscode-conformal && \
	  PAT="$${VSCE_PAT:-$$(cat $(PAT_FILE) 2>/dev/null)}" && \
	  { [ -n "$$PAT" ] || { echo "No PAT found. Put it in $(PAT_FILE) or export VSCE_PAT." >&2; exit 1; }; } && \
	  VSCE_PAT="$$PAT" npx @vscode/vsce verify-pat EthanDoughty

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
	docker run --rm -v "$$(pwd):/workspace" $(DOCKER_IMAGE):latest /workspace/$(FILE)
