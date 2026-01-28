#!/bin/sh
PROJECT=llama-stack-demo
APP_NAME=llama-stack-demo-lsd
# VALUES="--values intel.yaml"
# VALUES="--values nvidia.yaml"
# VALUES=""

helm template . --namespace ${PROJECT} --name-template ${APP_NAME} \
  --include-crds ${VALUES} 
  