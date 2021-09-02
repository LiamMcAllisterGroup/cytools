SHELL := /bin/bash

UNAME=$(shell uname)
ifneq (,$(findstring Darwin,$(UNAME)))
	machine=Mac
else ifneq (,$(findstring Linux,$(UNAME)))
	machine=Linux
else
	machine=UNKNOWN
endif

USERID=$(shell id -u)

.PHONY: all build rebuild install uninstall run pull update test

all:
	@echo "Please specify an instruction (e.g make build)"

build:
	@if [ "$(USERID)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		false; \
	fi
	@ echo "Building CYTools image for user $(USERID)..."
	sudo docker build -t cytools:uid-$(USERID) --build-arg USERID=$(USERID) .
	@ echo "Successfully build CYTools image for user $(USERID)"

rebuild:
	@if [ "$(USERID)" = "0" ]; then \
		echo "Please run make as a non-root user and without sudo!"; \
		false; \
	fi
	docker build --no-cache -t cytools:uid-$(USERID) --build-arg USERID=$(USERID) .

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

pull:
	git pull

update: pull install

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
