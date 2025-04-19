IMAGE_NAME ?= speechlab:latest
BASE_IMAGE ?= pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: docker-build docker-run docker-shell

docker-build:
	docker build \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		-t $(IMAGE_NAME) .

docker-run:
	docker run \
		--rm -it \
		--name speechlab \
		--gpus all \
		-p 5501:5501 \
		-p 5502:5502 \
		-p 5510:5510 \
		-p 5520:5520 \
		-v $(MAKEFILE_DIR)../data:/data \
		$(IMAGE_NAME) \
		--docker="True" \
		--data_dir="/data" \
		--fish_speech="fish_speech.cfg" \
		--xtts="none" \
		--giga_am="giga_am.cfg" \
		--ru_norm="ru_norm.cfg"

docker-run-all:
	docker run \
		--rm -it \
		--name speechlab \
		--gpus all \
		-p 5501:5501 \
		-p 5502:5502 \
		-p 5510:5510 \
		-p 5520:5520 \
		-v $(MAKEFILE_DIR)../data:/data \
		$(IMAGE_NAME) \
		--docker="True" \
		--data_dir="/data" \
		--fish_speech="fish_speech.cfg" \
		--xtts="xtts.cfg" \
		--giga_am="giga_am.cfg" \
		--ru_norm="ru_norm.cfg"

docker-shell:
	docker run --rm -it --entrypoint /bin/bash $(IMAGE_NAME)
