include build.env

APP_IMAGE=${DOCKER_REPO}/${APP_RELEASE}
BACKEND_IMAGE=${DOCKER_REPO}/${BACKEND_RELEASE}
FRONTEND_IMAGE=${DOCKER_REPO}/${FRONTEND_RELEASE}
DATASET_BROWSER_IMAGE=${DOCKER_REPO}/${DATASET_BROWSER_RELEASE}

app-build:
	docker build -f apps/app/Dockerfile -t ${APP_IMAGE}:${APP_VERSION} .

app-upload:
	docker push ${APP_IMAGE}:${APP_VERSION}

app-run:
	test -n "$(CV_PIPELINER_APP_CONFIG)"  # $$CV_PIPELINER_APP_CONFIG
	docker run -v /mnt/c/:/mnt/c/ -e CV_PIPELINER_APP_CONFIG=${CV_PIPELINER_APP_CONFIG} -p 80:80 -p 5000:5000 -t ${APP_IMAGE}:${APP_VERSION}

backend-build:
	docker build -f apps/backend/Dockerfile -t ${BACKEND_IMAGE}:${BACKEND_VERSION} .

backend-upload:
	docker push ${BACKEND_IMAGE}:${BACKEND_VERSION}

backend-run:
	test -n "$(CV_PIPELINER_BACKEND_MODEL_CONFIG)"  # $$CV_PIPELINER_BACKEND_MODEL_CONFIG
	docker run -v /mnt/c/:/mnt/c/ -e CV_PIPELINER_BACKEND_MODEL_CONFIG=${CV_PIPELINER_BACKEND_MODEL_CONFIG} -p 5000:5000 -t ${BACKEND_IMAGE}:${BACKEND_VERSION}

frontend-build:
	docker build -t ${FRONTEND_IMAGE}:${FRONTEND_VERSION} apps/frontend

frontend-run:
	docker container run -p 80:80 -t ${FRONTEND_IMAGE}:${FRONTEND_VERSION}

realtime-run: backend-run frontend-run

dataset-browser-build:
	docker build -f apps/dataset_browser/Dockerfile -t ${DATASET_BROWSER_IMAGE}:${DATASET_BROWSER_VERSION} .

dataset-browser-upload:
	docker push ${DATASET_BROWSER_IMAGE}:${DATASET_BROWSER_VERSION}

dataset-browser-run:
	test -n "$(CV_PIPELINER_DATASET_BROWSER_CONFIG)"  # $$CV_PIPELINER_DATASET_BROWSER_CONFIG
	docker run -v /mnt/c/:/mnt/c/ -e CV_PIPELINER_DATASET_BROWSER_CONFIG=${CV_PIPELINER_DATASET_BROWSER_CONFIG} -p 80:80 -p 81:81 -t ${DATASET_BROWSER_IMAGE}:${DATASET_BROWSER_VERSION}
