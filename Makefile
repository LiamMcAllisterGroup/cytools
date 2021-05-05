SHELL := /bin/bash

UNAME=$(shell uname)
ifneq (,$(findstring Darwin,$(UNAME)))
	machine=Mac
else ifneq (,$(findstring Linux,$(UNAME)))
	machine=Linux
else
	machine=UNKNOWN
endif

.PHONY: all build rebuild install uninstall run pull update

all:
	@echo "Please specify an instruction (e.g make build)"

build:
	docker build -t cytools .

rebuild:
	docker build --no-cache -t cytools .

install: build
	@if [ "$(machine)" = "Mac" ]; then \
		bash scripts/macos/install.sh; \
	else \
		bash scripts/linux/install.sh; \
	fi

uninstall:
	@if [ "$(machine)" = "Mac" ]; then \
		bash scripts/macos/uninstall.sh; \
	else \
		bash scripts/linux/uninstall.sh; \
	fi

run:
	@if [ "$(machine)" = "Mac" ]; then \
		bash scripts/macos/cytools; \
	else \
		bash scripts/linux/cytools; \
	fi

pull:
	git pull

update: pull install
