#!/bin/bash
#
# Uninstaller for the CYTools launcher/icon installed by install.sh (Linux).
# Removes the ~/.local launcher, icon, and desktop entry, then optionally
# removes the 'cytools' conda environment.
#
# Usage:  bash uninstall.sh

set -euo pipefail

ENV_NAME="cytools"
BIN_DIR="$HOME/.local/bin"
ICON_DIR="$HOME/.local/share/icons/hicolor/512x512/apps"
APP_DIR="$HOME/.local/share/applications"

echo "Removing CYTools launcher, icon, and menu entry..."
rm -fv "$BIN_DIR/cytools" \
       "$ICON_DIR/cytools.png" \
       "$APP_DIR/cytools.desktop"

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APP_DIR" || true
command -v gtk-update-icon-cache   >/dev/null 2>&1 && \
  gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true
echo "Launcher and icon removed."

# --- optionally remove the conda environment ---
CONDA=""
if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA="$CONDA_EXE"
else
  for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "/opt/conda"; do
    [[ -x "$c/bin/conda" ]] && { CONDA="$c/bin/conda"; break; }
  done
fi

if [[ -n "$CONDA" ]] && "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  read -r -p "Also remove the '$ENV_NAME' conda environment? [y/N] " reply
  if [[ "$reply" =~ ^[Yy]$ ]]; then
    "$CONDA" env remove -n "$ENV_NAME" -y
    echo "Removed the '$ENV_NAME' environment."
  else
    echo "Kept the '$ENV_NAME' environment (use it with: conda activate $ENV_NAME)."
  fi
fi

echo "Done."
