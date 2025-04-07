IMAGE_NAME ?= speechlab:latest
BASE_IMAGE ?= pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

.PHONY: docker-build docker-run docker-shell

docker-build:
	docker build \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		-t $(IMAGE_NAME) .

docker-run:
	docker run --rm -it $(IMAGE_NAME)

docker-shell:
	docker run --rm -it --entrypoint /bin/bash $(IMAGE_NAME)
