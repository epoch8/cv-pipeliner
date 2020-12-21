VERSION=0.4.0

APP_IMAGE=eu.gcr.io/e8-gke/cv-pipeliner-app:${VERSION}
FRONTEND_IMAGE=eu.gcr.io/e8-gke/cv-pipeliner-frontend:${VERSION}

build-be:
	docker build -t ${APP_IMAGE} -f Dockerfile .

build-fe:
	docker build -t ${FRONTEND_IMAGE} -f ./apps/frontend/Dockerfile ./apps/frontend/

build: build-be build-fe

upload-be:
	docker push ${APP_IMAGE}

upload-fe:
	docker push ${FRONTEND_IMAGE}

upload: upload-be upload-fe
