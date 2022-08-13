#!/bin/bash

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $dir/config.sh

if [ -z "$nb_port" ]; then
  echo "Notebook port not configured in $dir/config.sh"
  exit 1
fi

args="$EXPA_DOCKER_ARGS -p $nb_port:8888"
cmd="jupyter notebook --ip='0.0.0.0' $@"


EXPA_DOCKER_ARGS="$args" $dir/run.sh $cmd
