#!/bin/bash

rm -f /usr/local/bin/cytools
rm -f /usr/share/pixmaps/cytools.png
rm -f /usr/share/applications/cytools.desktop

read -p "Do you want to remove the CYTools Docker image? ([y]/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]
then
  exit
fi

docker rmi cytools
