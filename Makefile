SHELL := /bin/bash

UNAME=$(shell uname)
ifneq (,$(findstring Darwin,$(UNAME)))
	machine=Mac
else ifneq (,$(findstring Linux,$(UNAME)))
	machine=Linux
else
	machine=UNKNOWN
endif

UNAMEM=$(shell uname -m)
ifneq (,$(findstring arm64,$(UNAMEM)))
	arch=arm64
	aarch=aarch64
else
	arch=amd64
	aarch=x86_64
endif

USERID=$(shell id -u)
USERIDN=$(shell id -u -n)

.PHONY: all build install uninstall run test build-with-root-user

all:
	@ echo "Please specify an instruction (e.g make install)"

build:
	@ if [ "$(USERID)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		false; \
	fi
	@ echo "Deleting old CYTools image..."
	sudo docker rmi cytools:uid-$(USERID) | echo "Old CYTools image does not exist or cannot be deleted"
	@ echo "Building CYTools image for user $(USERIDN)..."
	sudo docker build --force-rm -t cytools:uid-$(USERID) --build-arg USERNAME=cytools\
	     --build-arg USERID=$(USERID) --build-arg ARCH=$(arch) --build-arg AARCH=$(aarch)\
			 --build-arg VIRTUAL_ENV=/home/cytools/cytools-venv/ --build-arg ALLOW_ROOT_ARG=" " .
	@ echo "Successfully built CYTools image for user $(USERIDN)"

install: build
	@if [ "$(USERID)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		false; \
	fi
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

uninstall:
	@if [ "$(USERID)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		false; \
	fi
	@if [ "$(machine)" = "Mac" ]; then \
		sudo rm -rf /Applications/CYTools.app/; \
		sudo rm -f /usr/local/bin/cytools; \
	else \
		sudo rm -f /usr/local/bin/cytools; \
		sudo rm -f /usr/share/pixmaps/cytools.png; \
		sudo rm -f /usr/share/applications/cytools.desktop; \
	fi
	sudo docker rmi cytools:uid-$(USERID)

run:
	@if [ "$(USERID)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		false; \
	fi
	@if [ "$(machine)" = "Mac" ]; then \
		bash scripts/macos/cytools; \
	else \
		bash scripts/linux/cytools; \
	fi

unittests:
	@if [ "$(USERID)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		false; \
	fi
	wget "https://github.com/LiamMcAllisterGroup/cytools/releases/download/v0.0.1/unittests.tar.gz"
	tar zxvf unittests.tar.gz
	rm unittests.tar.gz

test: unittests
	@if [ "$(USERID)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		false; \
	fi
	sudo docker run --rm -it -v ${PWD}/unittests:/home/cytools/mounted_volume/ cytools bash -c "bash run_tests.sh"

build-with-root-user:
	@ echo " "
	@ echo "********************************************************************"
	@ echo "Warning: You are building an image with a root user. Any user with "
	@ echo "access to this image will be able to have root access to the host "
	@ echo "computer as well. Please proceed with care.";
	@ echo "********************************************************************"
	@ echo " "
	@ read -p "Press enter to continue or ctrl+c to cancel"
	@ echo "Deleting old CYTools image..."
	sudo docker rmi cytools:root | echo "Old CYTools image does not exist or cannot be deleted"
	@ echo "Building CYTools image for root user..."
	sudo docker build -t cytools:root --build-arg USERNAME=root\
	     --build-arg USERID=0 --build-arg ARCH=$(arch) --build-arg AARCH=$(aarch)\
			 --build-arg VIRTUAL_ENV=/opt/cytools/cytools-venv/ --build-arg ALLOW_ROOT_ARG="--allow-root" .
	@ echo "Successfully built CYTools image with root user."
