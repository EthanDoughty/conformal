#!/bin/bash
# Build self-contained native binaries for all platforms.
# Output goes to dist/ directory.
#
# Usage: ./scripts/build-releases.sh
# Requires: .NET 8 SDK with AOT workloads
#
# Note: cross-compilation (building win-x64 on Linux) requires
# the appropriate cross-compilation toolchain. For CI, build
# each platform on its native runner.

set -e

VERSION="${1:-3.8.0}"
PROJECT="src/analyzer/ConformalAnalyzer.fsproj"
DIST="dist"

rm -rf "$DIST"
mkdir -p "$DIST"

publish_target() {
    local rid="$1"
    local ext="$2"
    echo "Building $rid..."
    dotnet publish "$PROJECT" \
        -c Release \
        -r "$rid" \
        --self-contained \
        -p:NuGetAudit=false \
        -o "$DIST/$rid" \
        2>&1 | tail -1

    local binary="$DIST/$rid/conformal-parse${ext}"
    if [ -f "$binary" ]; then
        local size=$(du -h "$binary" | cut -f1)
        echo "  $rid: $binary ($size)"

        # Package as tar.gz (Unix) or zip (Windows)
        pushd "$DIST/$rid" > /dev/null
        if [ "$ext" = ".exe" ]; then
            zip -q "../conformal-${VERSION}-${rid}.zip" "conformal-parse${ext}"
        else
            tar czf "../conformal-${VERSION}-${rid}.tar.gz" "conformal-parse${ext}"
        fi
        popd > /dev/null
    else
        echo "  FAILED: $binary not found"
    fi
}

# Build for the current platform only (cross-compile needs native toolchains)
# In CI, each platform runner calls this with its own RID.
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  publish_target "linux-x64" "" ;;
    Linux-aarch64) publish_target "linux-arm64" "" ;;
    Darwin-x86_64) publish_target "osx-x64" "" ;;
    Darwin-arm64)  publish_target "osx-arm64" "" ;;
    MINGW*|MSYS*)  publish_target "win-x64" ".exe" ;;
    *)
        echo "Unknown platform, building linux-x64 as default"
        publish_target "linux-x64" ""
        ;;
esac

echo ""
echo "Release artifacts:"
ls -lh "$DIST"/*.tar.gz "$DIST"/*.zip 2>/dev/null
