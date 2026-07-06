#!/bin/bash
#
# One-stop CYTools installer (macOS) for users who just want a clickable app.
# It will, in order:
#   1. install Miniforge (conda) if none is found            (asks first)
#   2. create or update the 'cytools' conda environment
#   3. assemble CYTools.app into ~/Applications (no admin required)
#
# Usage:  bash install.sh

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/macos
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/environment.yml"
ENV_NAME="cytools"

# --- 0. platform preflight: Intel Macs are unsupported (native deps are arm64-only) ---
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" && -z "${CYTOOLS_FORCE_INTEL:-}" ]]; then
  cat >&2 << 'EOF'
Error: This looks like an Intel (x86_64) Mac, which CYTools does not support.

Several of CYTools' native dependencies (cygv, triangulumancer, pypalp) publish
macOS wheels for Apple Silicon (arm64) only, so the install cannot complete here.

Supported options:
  * an Apple Silicon Mac (M-series)
  * a Linux machine (or WSL on Windows)

If you ARE on Apple Silicon but see this, you're likely in an x86_64 / Rosetta
shell or conda -- use a native arm64 terminal and conda instead.

(To try anyway, e.g. to build from source, re-run with CYTOOLS_FORCE_INTEL=1.)
EOF
  exit 1
fi

# --- 1. locate conda, or offer to install Miniforge --------------------------
find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then echo "$CONDA_EXE"; return 0; fi
  local c
  for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
           "/opt/miniconda3" "/opt/anaconda3" "/opt/homebrew/Caskroom/miniforge/base"; do
    [[ -x "$c/bin/conda" ]] && { echo "$c/bin/conda"; return 0; }
  done
  return 1
}

install_miniforge() {                 # sets global CONDA, or exits
  local prefix="$HOME/miniforge3" arch asset url dl tmp reply
  arch="$(uname -m)"                  # arm64 or x86_64 on macOS
  asset="Miniforge3-MacOSX-${arch}.sh"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/$asset"

  echo "No conda installation was found." >&2
  echo "This will download and install Miniforge (conda) to: $prefix" >&2
  read -r -p "Proceed with installing Miniforge? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted. Install conda manually and re-run." >&2; exit 1; }

  if   command -v curl >/dev/null 2>&1; then dl=(curl -fsSL -o)
  elif command -v wget >/dev/null 2>&1; then dl=(wget -qO)
  else echo "Need curl or wget to download Miniforge." >&2; exit 1; fi

  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
  echo "Downloading $asset ..." >&2
  "${dl[@]}" "$tmp/$asset" "$url"
  echo "Installing Miniforge to $prefix ..." >&2
  bash "$tmp/$asset" -b -p "$prefix"
  CONDA="$prefix/bin/conda"
}

CONDA=""
if path="$(find_conda)"; then CONDA="$path"; else install_miniforge; fi
echo "Using conda at: $CONDA"

# --- 2. create or update the 'cytools' environment ---------------------------
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing '$ENV_NAME' environment..."
  "$CONDA" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune || {
    echo "Error: environment update failed (see the conda/pip output above)." >&2; exit 1; }
else
  echo "Creating '$ENV_NAME' environment (this can take several minutes)..."
  "$CONDA" env create -f "$ENV_FILE" || {
    echo "Error: environment creation failed (see the conda/pip output above)." >&2; exit 1; }
fi

# --- 3. assemble CYTools.app into ~/Applications -----------------------------
APP="$HOME/Applications/CYTools.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

install -m 0755 "$SRC_DIR/cytools"      "$APP/Contents/MacOS/cytools"
install -m 0755 "$SRC_DIR/launcher.sh"  "$APP/Contents/MacOS/launcher.sh"
install -m 0755 "$SRC_DIR/CYToolsApp"   "$APP/Contents/MacOS/CYToolsApp"
install -m 0644 "$SRC_DIR/info.plist"   "$APP/Contents/Info.plist"
install -m 0644 "$SRC_DIR/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

# nudge LaunchServices/Finder to notice the new bundle and its icon
touch "$APP"
if [[ -x /System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister ]]; then
  /System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister -f "$APP" || true
fi

echo
echo "Done! CYTools.app is installed in ~/Applications."
echo "Open it from Launchpad, Spotlight, or Finder (~/Applications)."

if [ -e /Applications/CYTools.app ]; then
  echo
  echo "Warning: an older /Applications/CYTools.app exists and may open instead of this one." >&2
  echo "Remove it with:  sudo rm -rf /Applications/CYTools.app" >&2
fi
