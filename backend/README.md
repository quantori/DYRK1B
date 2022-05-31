## Development environment setup

Note: This application (build and run process) is adapted for Linux. Operation on other OS's is not guaranteed. Please use virtualized or containerized Linux environment.

### Deploying a local environment using Docker       

0 - Following requirements should be satisfied:
* Docker + Docker Compose
* GNU Make

1 - Get your own `.env` and `docker-compose.yaml` files:
```shell
cp .sample.env .env
cp docker-compose.local.sample.yml docker-compose.local.yml
```

2 - Build `qsar-be` image:
```shell
docker-compose --file docker-compose.local.yml build qsar-be
```

[//]: # (3 - Seed databases with datasets:)

[//]: # (```shell)

[//]: # (make docker/async/data)

[//]: # (```)
Note: First time usage may take ~ 20 minutes.

3 - To start everything:
```shell
docker-compose --file docker-compose.local.yml up --detach qsar-be
```

4 - After completion of your work, to stop everything:
```shell
docker-compose --file docker-compose.local.yml down
```