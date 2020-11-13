VERSION=0.1.0
IMAGE=epoch8/cv-pipeliner:${VERSION}

build:
	docker build -t ${IMAGE} -f Dockerfile .