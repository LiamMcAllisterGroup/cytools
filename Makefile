SHELL := /bin/bash

# Get machine type
uname := $(shell uname)
machine := $(if $(filter Darwin,$(uname)),Mac,$(if $(filter Linux,$(uname)),Linux,UNKNOWN))

# Get machine architecture
uname_m := $(shell uname -m)
arch := $(if $(filter arm64,$(uname_m)),arm64,amd64)
aarch := $(if $(filter arm64,$(uname_m)),aarch64,x86_64)

# Get user information
userid := $(shell id -u)
userid_n := $(shell id -u -n)

# Argument to install optional packages
optional_pkgs ?= 0

.PHONY: all build install uninstall run test build-with-root-user check-not-root-user get-sudo-credentials

all:
	@echo "Please specify an instruction (e.g make install)."

build: check-not-root-user get-sudo-credentials
	@ echo "Deleting old CYTools image..."
	sudo docker rmi cytools:uid-$(userid) || echo "Old CYTools image does not exist or cannot be deleted"
	@echo "Building CYTools image for user $(userid_n)..."
	sudo docker pull python:3.11-bullseye
	sudo docker build --no-cache --force-rm -t cytools:uid-$(userid) \
		--build-arg USERNAME=cytools --build-arg USERID=$(userid) \
		--build-arg ARCH=$(arch) --build-arg AARCH=$(aarch) \
		--build-arg VIRTUAL_ENV=/home/cytools/cytools-venv/ \
		--build-arg ALLOW_ROOT_ARG=" " \
		--build-arg OPTIONAL_PKGS=$(optional_pkgs) \
		--build-arg PORT_ARG=$$(( $(userid) + 2875 )) .

	@echo "Successfully built CYTools image for user $(userid_n)"

build-fast: check-not-root-user get-sudo-credentials
	@echo "Building CYTools image for user $(userid_n)..."
	sudo docker pull python:3.11-bullseye
	sudo docker build -t cytools:uid-$(userid) \
		--build-arg USERNAME=cytools --build-arg USERID=$(userid) \
		--build-arg ARCH=$(arch) --build-arg AARCH=$(aarch) \
		--build-arg VIRTUAL_ENV=/home/cytools/cytools-venv/ \
		--build-arg ALLOW_ROOT_ARG=" " \
		--build-arg OPTIONAL_PKGS=$(optional_pkgs) \
		--build-arg PORT_ARG=$$(( $(userid) + 2875 )) .

	@echo "Successfully built CYTools image for user $(userid_n)"

install: build
	@echo "Copying launcher script and associated files..."
	@if [ "$(machine)" = "Mac" ]; then \
		sudo cp scripts/macos/cytools /usr/local/bin/cytools; \
		sudo chmod +x /usr/local/bin/cytools; \
		sudo mkdir -p /Applications/CYTools.app/Contents/MacOS/; \
		sudo cp scripts/macos/info.plist /Applications/CYTools.app/Contents/info.plist; \
		sudo cp scripts/macos/CYToolsApp /Applications/CYTools.app/Contents/MacOS/CYToolsApp; \
		sudo chmod +x /Applications/CYTools.app/Contents/MacOS/CYToolsApp; \
		sudo cp scripts/macos/launcher.sh /Applications/CYTools.app/Contents/MacOS/launcher.sh; \
		sudo chmod +x /Applications/CYTools.app/Contents/MacOS/launcher.sh; \
		sudo mkdir -p /Applications/CYTools.app/Contents/Resources/; \
		sudo cp scripts/macos/AppIcon.icns /Applications/CYTools.app/Contents/Resources/AppIcon.icns; \
	else \
		sudo cp scripts/linux/cytools /usr/local/bin/cytools; \
		sudo chmod +x /usr/local/bin/cytools; \
		sudo cp scripts/linux/cytools.png /usr/share/pixmaps/cytools.png; \
		sudo cp scripts/linux/cytools.desktop /usr/share/applications/cytools.desktop; \
	fi

	@echo "Installation finished successfully!"

uninstall: check-not-root-user
	@if [ "$(machine)" = "Mac" ]; then \
		sudo rm -rf /Applications/CYTools.app/; \
		sudo rm -f /usr/local/bin/cytools; \
	else \
		sudo rm -f /usr/local/bin/cytools; \
		sudo rm -f /usr/share/pixmaps/cytools.png; \
		sudo rm -f /usr/share/applications/cytools.desktop; \
	fi
	sudo docker rmi cytools:uid-$(userid) || true

run: check-not-root-user
	@if [ "$(machine)" = "Mac" ]; then \
		bash scripts/macos/cytools; \
	else \
		bash scripts/linux/cytools; \
	fi

test: check-not-root-user
	sudo docker run --rm -it cytools:uid-$(userid) bash -c "cd /opt/cytools/unittests/; bash /opt/cytools/unittests/run_tests.sh"

build-with-root-user:
	@ echo " "
	@ echo "********************************************************************"
	@ echo "Warning: You are building an image with a root user. Any user with "
	@ echo "access to this image will be able to have root access to the host "
	@ echo "computer as well. Please proceed with care.";
	@ echo "********************************************************************"
	@ echo " "
	@ read -p "Press enter to continue or ctrl+c to cancel"
	@echo "Deleting old CYTools image..."
	sudo docker rmi cytools:root || echo "Old CYTools image does not exist or cannot be deleted"
	@echo "Building CYTools image for root user..."
	sudo docker pull python:3.11-bullseye
	sudo docker build -t cytools:root \
		--build-arg USERNAME=root --build-arg USERID=0 \
		--build-arg ARCH=$(arch) --build-arg AARCH=$(aarch) \
		--build-arg VIRTUAL_ENV=/opt/cytools/cytools-venv/ \
		--build-arg ALLOW_ROOT_ARG="--allow-root" \
		--build-arg OPTIONAL_PKGS=$(optional_pkgs) \
		--build-arg PORT_ARG=2875 .

	@echo "Successfully built CYTools image with root user."

check-not-root-user:
	@if [ "$(userid)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		exit 1; \
	fi

get-sudo-credentials:
	@ echo "Building a Docker image requires sudo privileges. Please enter your password:"
	@ sudo echo ""

.PHONY: all build install uninstall run test build-with-root-user check-not-root-user get-sudo-credentials
