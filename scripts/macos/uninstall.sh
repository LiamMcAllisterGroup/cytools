#!/bin/bash

rm -rf /Applications/CYTools.app/
rm -f /usr/local/bin/cytools

read -p "Do you want to remove the CYTools Docker image? ([y]/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]
then
  exit
fi

docker rmi cytools
