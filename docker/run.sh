#!/bin/bash

# Get current dir and source config
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
repo_dir="$dir/.."
source $dir/config.sh

# Get extra docker args specified by caller
docker_args=$EXPA_DOCKER_ARGS

# Don't run as root unless configured
if [ "$as_root" != "1" ]; then
  docker_args="$docker_args -u $(id -u):$(id -g)"
fi

# Maybe add gpus
if [ ! -z "$gpus" ]; then
  docker_args="$docker_args --gpus $gpus"
fi
if [ ! -z "$cuda_devices" ]; then
  docker_args="$docker_args -e CUDA_VISIBLE_DEVICES=$cuda_devices"
fi

# Generate docker command
docker run --rm -it \
  -v "$repo_dir:/workspace" \
  -e "WANDB_API_KEY=$wandb_key" \
  -e "WANDB_MODE=$wandb_mode" \
  $docker_args $image $@
