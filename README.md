# Two-stage pipeliner
- Detection
- Classification

# App
How to build and run locally:

- Set the config `apps/config.yaml`, then note it in `docker-compose.yaml`
```
make build
docker-compose run
```

# Zipkin
docker run --rm -p 9411:9411 openzipkin/zipkin