#!/bin/bash
# Install CYTools in a new conda environment named "cytools".
#
# Usage: curl -fsSL https://cy.tools/install.sh | bash
set -eo pipefail  # exit immediately on error

# Check that conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not on your PATH." >&2
    echo "Please install conda before running this script. We recommend Miniforge:" >&2
    echo "https://conda-forge.org/download/" >&2
    exit 1
fi

# Check that the environment doesn't already exist
if conda env list | grep -qE "^cytools\s"; then
    echo "Error: a conda environment named 'cytools' already exists." >&2
    echo "To update CYTools, run 'conda activate cytools' and then 'pip install --upgrade cytools'." >&2
    echo "To reinstall it, first remove the environment with 'conda env remove --name cytools'." >&2
    exit 1
fi

# Fetch the environment file into a temporary directory that is removed on exit
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
curl -fsSL -o "$tmp_dir/environment.yml" https://raw.githubusercontent.com/LiamMcAllisterGroup/cytools/refs/heads/main/environment.yml

# Create the conda environment
conda env create -f "$tmp_dir/environment.yml"
