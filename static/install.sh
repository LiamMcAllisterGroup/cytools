#!/bin/bash
set -e  # exit immediately on error

# Check that condais installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not on your PATH." >&2
    echo "Please install Miniconda or Anaconda before running this script." >&2
    exit 1
fi

# Make temporary director for use in the install
cd /tmp/
tmp_dir="cytools-update-$RANDOM"
mkdir $tmp_dir
cd $tmp_dir

# Fetch the environment file
curl -fsSL -o environment.yml https://raw.githubusercontent.com/LiamMcAllisterGroup/cytools/refs/heads/main/environment.yml

# Install the conda environment
conda env create -f environment.yml

# Cleanup
cd /tmp/
rm -rf cytools-update-*
