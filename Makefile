VERSION=0.7.0

# APP_IMAGE=eu.gcr.io/e8-gke/cv-pipeliner-app:${VERSION}
APP_IMAGE=dcr.epoch8.co/cv-pipeliner-app:${VERSION}

build:
	docker build -t ${APP_IMAGE} -f Dockerfile .

upload:
	docker push ${APP_IMAGE}
