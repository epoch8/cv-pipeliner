include build.env

APP_IMAGE=${DOCKER_REPO}/${APP_RELEASE}
BACKEND_IMAGE=${DOCKER_REPO}/${BACKEND_RELEASE}

app-build:
	docker build -f app/DockerFile -t ${APP_IMAGE}:${APP_VERSION} .

app-upload:
	docker push ${APP_IMAGE}:${APP_VERSION}

app-run:
	test -n "$(CV_PIPELINER_APP_CONFIG)"  # $$CV_PIPELINER_APP_CONFIG
	docker run -e CV_PIPELINER_APP_CONFIG=${CV_PIPELINER_APP_CONFIG} -p 80:80 -t ${APP_IMAGE}:${APP_VERSION}

backend-build:
	docker build -f backend/DockerFile -t ${BACKEND_RELEASE}:${BACKEND_VERSION} .

backend-upload:
	docker push ${BACKEND_RELEASE}:${BACKEND_VERSION}

backend-run:
	docker run -p 5000:5000 -t ${BACKEND_RELEASE}:${BACKEND_VERSION}
