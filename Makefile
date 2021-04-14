VERSION=0.6.1-tracing16

APP_IMAGE=dcr.epoch8.co/cv-pipeliner-app:${VERSION}

build:
	docker build -t ${APP_IMAGE} -f Dockerfile .

upload:
	docker push ${APP_IMAGE}
