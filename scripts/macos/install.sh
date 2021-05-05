#!/bin/bash

cp scripts/macos/cytools /usr/local/bin/cytools
chmod +x /usr/local/bin/cytools
mkdir -p /Applications/CYTools.app/Contents/MacOS/
cp scripts/macos/info.plist /Applications/CYTools.app/Contents/info.plist
cp scripts/macos/CYToolsApp /Applications/CYTools.app/Contents/MacOS/CYToolsApp
chmod +x /Applications/CYTools.app/Contents/MacOS/CYToolsApp
cp scripts/macos/launcher.sh /Applications/CYTools.app/Contents/MacOS/launcher.sh
chmod +x /Applications/CYTools.app/Contents/MacOS/launcher.sh
mkdir -p /Applications/CYTools.app/Contents/Resources/
cp scripts/macos/AppIcon.icns /Applications/CYTools.app/Contents/Resources/AppIcon.icns
