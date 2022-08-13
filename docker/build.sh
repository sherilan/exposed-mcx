#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"
source config.sh
docker build -t $image .
