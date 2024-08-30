SHELL := /bin/bash

# Determine the operating system of the host machine
UNAME := $(shell uname)
MACHINE := $(if $(filter Darwin,$(UNAME)),Mac,$(if $(filter Linux,$(UNAME)),Linux,UNKNOWN))

# Determine the machine architecture
UNAME_M := $(shell uname -m)
ARCH := $(if $(filter arm64,$(UNAME_M)),arm64,amd64)
AARCH := $(if $(filter arm64,$(UNAME_M)),aarch64,x86_64)

# Get the current user's ID and name
USERID := $(shell id -u)
USERID_N := $(shell id -u -n)

# Option to install additional packages, set to 0/false by default
OPTIONAL_PKGS ?= 0

# Option to install Macaulay2
INSTALL_M2 ?= 0

# Option to install Sage
INSTALL_SAGE ?= 0

# Default build type is set to 'build'
BUILD_TYPE := build

# Define phony targets to prevent conflicts with files of the same name
.PHONY: all build-common build build-fast build-with-root-user install uninstall run test check-not-root-user get-sudo-credentials

# Default target; displays a message to the user
all:
	@echo "Please specify an instruction (e.g make install)."

# Check if the current user is root and exit if true
check-not-root-user:
	@if [ "$(USERID)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		exit 1; \
	fi

# Request sudo credentials from the user
get-sudo-credentials:
	@sudo -n true 2>/dev/null || { \
        echo -n "Building a Docker image requires sudo privileges. "; \
        sudo -v; \
    }

# Common build steps for non-root build types
build-common:
	@echo "Building CYTools image for user $(USERID_N)..."
	@{ sudo docker pull ubuntu:noble && \
	   sudo docker build $(DOCKER_BUILD_OPTS) -t cytools:uid-$(USERID) \
		--build-arg USERNAME=cytools --build-arg USERID=$(USERID) \
		--build-arg ARCH=$(ARCH) --build-arg AARCH=$(AARCH) \
		--build-arg VIRTUAL_ENV=/home/cytools/cytools-venv/ \
		--build-arg ALLOW_ROOT_ARG=" " \
		--build-arg OPTIONAL_PKGS=$(OPTIONAL_PKGS) \
		--build-arg INSTALL_M2=$(INSTALL_M2) \
		--build-arg INSTALL_SAGE=$(INSTALL_SAGE) \
		--build-arg PORT_ARG=$$(( $(USERID) + 2875 )) .; } > build.log
	@echo "Successfully built CYTools image for user $(USERID_N)"

# Standard build process
build: DOCKER_BUILD_OPTS=--no-cache --force-rm
build: check-not-root-user get-sudo-credentials
	@echo -n "Deleting old CYTools image... "
	@sudo docker rmi cytools:uid-$(USERID) > /dev/null 2>&1 && echo "done!" || echo "old CYTools image does not exist or cannot be deleted..."
	@$(MAKE) -C . --no-print-directory build-common DOCKER_BUILD_OPTS='$(DOCKER_BUILD_OPTS)'

# Fast build process, allows cached info for quicker build
build-fast: DOCKER_BUILD_OPTS=
build-fast: check-not-root-user get-sudo-credentials
	@$(MAKE) -C . --no-print-directory build-common DOCKER_BUILD_OPTS='$(DOCKER_BUILD_OPTS)'

# Build process when running as root user
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
	@sudo docker rmi cytools:root > /dev/null 2>&1 && echo "done!" || echo "old CYTools image does not exist or cannot be deleted..."
	@echo "Building CYTools image for root user..."
	@{ sudo docker pull ubuntu:noble && \
	   sudo docker build -t cytools:root \
		--build-arg USERNAME=root --build-arg USERID=0 \
		--build-arg ARCH=$(ARCH) --build-arg AARCH=$(AARCH) \
		--build-arg VIRTUAL_ENV=/opt/cytools/cytools-venv/ \
		--build-arg ALLOW_ROOT_ARG="--allow-root" \
		--build-arg OPTIONAL_PKGS=$(OPTIONAL_PKGS) \
		--build-arg INSTALL_M2=$(INSTALL_M2) \
		--build-arg INSTALL_SAGE=$(INSTALL_SAGE) \
		--build-arg PORT_ARG=2875 .; } > build.log
	@echo "Successfully built CYTools image with root user."

# Installation process
install: $(BUILD_TYPE)
	@echo "Copying launcher script and associated files..."
	@if [ "$(MACHINE)" = "Mac" ]; then \
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

# Uninstallation process
uninstall: check-not-root-user
	@if [ "$(MACHINE)" = "Mac" ]; then \
		sudo rm -rf /Applications/CYTools.app/; \
		sudo rm -f /usr/local/bin/cytools; \
	else \
		sudo rm -f /usr/local/bin/cytools; \
		sudo rm -f /usr/share/pixmaps/cytools.png; \
		sudo rm -f /usr/share/applications/cytools.desktop; \
	fi
	@sudo docker rmi cytools:uid-$(USERID) || true

# Test the application
test: check-not-root-user
	sudo docker run --rm -it cytools:uid-$(USERID) bash -c "cd /opt/cytools/unittests/; bash /opt/cytools/unittests/run_tests.sh"
