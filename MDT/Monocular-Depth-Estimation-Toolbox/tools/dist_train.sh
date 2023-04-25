#!/usr/bin/env bash

echo $0
echo $1
echo $2

CONFIG=$1
GPUS=$2
PORT=${PORT:-38423}
echo $(dirname "$0")
echo $PORT
echo ${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

#python $(dirname "$0")/train.py $CONFIG --launcher none ${@:3}
