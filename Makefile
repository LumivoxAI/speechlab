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
		-p 5503:5503 \
		-v $(MAKEFILE_DIR)../model:/model \
		-v $(MAKEFILE_DIR)../reference:/reference \
		$(IMAGE_NAME)

docker-shell:
	docker run --rm -it --entrypoint /bin/bash $(IMAGE_NAME)
