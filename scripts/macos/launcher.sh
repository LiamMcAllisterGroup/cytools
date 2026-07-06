#!/bin/bash
#
# Bundled launcher: runs the CYTools launcher that sits next to it in the
# app bundle (Contents/MacOS/cytools).

exec "$(cd "$(dirname "$0")" && pwd)/cytools"
