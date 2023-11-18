#!/bin/bash
dir_to_mount=$(pwd)
docker run -it -v $HOME/.cache:/root/.cache -v $dir_to_mount:/workspace shaunstoltz/deepspeed:latest
