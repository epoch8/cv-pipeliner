include build.env

IMAGE=${DOCKER_REPO}/${RELEASE}

build:
	docker build -t ${IMAGE}:${VERSION} .

upload:
	docker push ${IMAGE}:${VERSION}

run:
	ifndef TWO_STAGE_PIPELINER_APP_CONFIG:
	$(error Variable TWO_STAGE_PIPELINER_APP_CONFIG is not set)
	endif
	docker run -e TWO_STAGE_PIPELINER_APP_CONFIG=${TWO_STAGE_PIPELINER_APP_CONFIG} -p 80:80 -t ${IMAGE}:${VERSION}

