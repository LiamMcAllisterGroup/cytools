#!/bin/bash
#
# Uninstaller for the CYTools app installed by install.sh (macOS).
# Removes ~/Applications/CYTools.app, then optionally removes the 'cytools'
# conda environment.
#
# Usage:  bash uninstall.sh

set -euo pipefail

ENV_NAME="cytools"
APP="$HOME/Applications/CYTools.app"

echo "Removing CYTools.app..."
rm -rfv "$APP"

# tell LaunchServices to forget the bundle
LSREG=/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister
[[ -x "$LSREG" ]] && "$LSREG" -u "$APP" 2>/dev/null || true
echo "CYTools.app removed from ~/Applications."

if [ -e /Applications/CYTools.app ]; then
  echo "Note: an older /Applications/CYTools.app also exists; remove it with:"
  echo "  sudo rm -rf /Applications/CYTools.app"
fi

# optionally remove the conda environment
# ---------------------------------------
CONDA=""
if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA="$CONDA_EXE"
else
  for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
           "/opt/miniconda3" "/opt/anaconda3" "/opt/homebrew/Caskroom/miniforge/base"; do
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
