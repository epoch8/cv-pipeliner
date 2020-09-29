include build.env

IMAGE=${DOCKER_REPO}/${RELEASE}

build:
	docker build -t ${IMAGE}:${VERSION} .

upload:
	docker push ${IMAGE}:${VERSION}

run:
ifndef CV_PIPELINER_APP_CONFIG
$(error Variable CV_PIPELINER_APP_CONFIG is not set)
endif
	docker run -e CV_PIPELINER_APP_CONFIG=${CV_PIPELINER_APP_CONFIG} -p 80:80 -t ${IMAGE}:${VERSION}

