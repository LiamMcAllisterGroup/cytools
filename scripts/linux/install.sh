#!/bin/bash

# Check if a CYTools image already exists
images=$(docker images -a -q cytools)
if [[ "$images" == "" ]]
then
  exists=false
else
  exists=true
fi

cp scripts/linux/cytools /usr/local/bin/cytools
chmod +x /usr/local/bin/cytools
cp scripts/linux/cytools.png /usr/share/pixmaps/cytools.png
cp scripts/linux/cytools.desktop /usr/share/applications/cytools.desktop

# If it was previously installed then it doesn't tell people to add the users
# to the Docker group
if $exists
then
  exit
fi

echo ""
echo "To use the launcher script without sudo the users need to be part of the docker group."
echo "Warning: The docker group gives privileges equivalent to the root user."
read -p "Do you want to add all users to the docker group? ([y]/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]
then
  echo ""
  echo "You will need to manually add users to the docker group to use the launcher script without sudo."
  echo "You can do so with the following commands:"
  echo "sudo groupadd docker"
  echo "sudo usermod -aG docker \$USER"
  echo ""
  exit
fi

# Add users to docker group if necessary
groupadd docker
for ID in $(cat /etc/passwd | grep /home | cut -d ':' -f1)
do
  if id -nGz "$ID" | grep -qzxF docker
  then
    echo ""
    echo "User '$ID' already belongs to docker group"
  else
    echo ""
    echo "Adding '$ID' to docker group"
    sudo usermod -aG docker $ID
    echo "****************************************************"
    echo "Note: You will need to reboot your computer before "
    echo "you can use the CYTools launcher script."
    echo "****************************************************"
  fi
done
