VERSION=0.6.1

APP_IMAGE=eu.gcr.io/e8-gke/cv-pipeliner-app:${VERSION}

build:
	docker build -t ${APP_IMAGE} -f Dockerfile .

upload:
	docker push ${APP_IMAGE}
