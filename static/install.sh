#!/bin/bash

cd /tmp/
tmp_dir="cytools-update-$RANDOM"
mkdir $tmp_dir
cd $tmp_dir

# Fetch the latest release tarball
curl -s https://api.github.com/repos/LiamMcAllisterGroup/cytools/tags \
| grep "tarball_url" \
| grep -Eo 'https://[^\"]*' \
| sed -n '1p' \
| xargs wget -q -O - \
| tar -xz --strip-components 1

# Default value of whether to install optional packages
: "${OPTIONAL_PKGS:=0}"

# Install
make install OPTIONAL_PKGS=$OPTIONAL_PKGS

cd /tmp/
rm -rf cytools-update-*
