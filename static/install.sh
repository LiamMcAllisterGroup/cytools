#!/bin/bash

cd /tmp/
tmp_dir="cytools-update-$RANDOM"
mkdir $tmp_dir
cd $tmp_dir
curl -s https://api.github.com/repos/LiamMcAllisterGroup/cytools/tags | grep "tarball_url" | grep -Eo 'https://[^\"]*' | sed -n '1p' | xargs wget -q -O - | tar -xz --strip-components 1
make install
cd /tmp/
rm -rf cytools-update-*
