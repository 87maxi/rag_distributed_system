#!/bin/bash

git clone  https://github.com/docker/model-runner.git

cd model-runner/cmd/cli

make build

cp  ~/model-runner/cmd/cli/model-cli  ~/.docker/cli-plugins/docker-model


cd ~ && git clone https://github.com/docker/mcp-gateway.git

cd mcp-gateway && make docker-mcp 
