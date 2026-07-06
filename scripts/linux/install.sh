#!/bin/bash
#
# One-stop CYTools installer (Linux) for users who just want a clickable app.
# It will, in order:
#   1. install Miniforge (conda) if none is found            (asks first)
#   2. create or update the 'cytools' conda environment
#   3. install the CYTools launcher + app-menu icon into ~/.local (no sudo)
#
# Usage:  bash install.sh

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/linux
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/environment.yml"
ENV_NAME="cytools"

# --- 1. locate conda, or offer to install Miniforge --------------------------
find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then echo "$CONDA_EXE"; return 0; fi
  local c
  for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "/opt/conda"; do
    [[ -x "$c/bin/conda" ]] && { echo "$c/bin/conda"; return 0; }
  done
  return 1
}

install_miniforge() {                 # sets global CONDA, or exits
  local prefix="$HOME/miniforge3" arch asset url dl tmp reply
  arch="$(uname -m)"
  case "$arch" in
    x86_64)  asset="Miniforge3-Linux-x86_64.sh" ;;
    aarch64) asset="Miniforge3-Linux-aarch64.sh" ;;
    *) echo "Unsupported architecture '$arch'; please install conda manually." >&2; exit 1 ;;
  esac
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
  "$CONDA" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "Creating '$ENV_NAME' environment (this can take several minutes)..."
  "$CONDA" env create -f "$ENV_FILE"
fi

# --- 3. install launcher + icon + desktop entry into ~/.local ----------------
BIN_DIR="$HOME/.local/bin"
ICON_DIR="$HOME/.local/share/icons/hicolor/512x512/apps"
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$BIN_DIR" "$ICON_DIR" "$APP_DIR"

install -m 0755 "$SRC_DIR/cytools"     "$BIN_DIR/cytools"
install -m 0644 "$SRC_DIR/cytools.png" "$ICON_DIR/cytools.png"
sed -e "s|^Exec=.*|Exec=$BIN_DIR/cytools|" -e "s|^Icon=.*|Icon=$ICON_DIR/cytools.png|" \
    "$SRC_DIR/cytools.desktop" > "$APP_DIR/cytools.desktop"

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APP_DIR" || true
command -v gtk-update-icon-cache   >/dev/null 2>&1 && \
  gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true

echo
echo "Done! CYTools should now appear in your application menu."
echo "You can also run 'cytools' from a new terminal."
echo ":$PATH:" | grep -q ":$BIN_DIR:" || echo "(If 'cytools' isn't found, add ~/.local/bin to your PATH.)"
