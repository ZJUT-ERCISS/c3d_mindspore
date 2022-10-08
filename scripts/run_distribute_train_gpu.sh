#!/bin/bash
# bash run_standalone_train_gpu.sh [CONFIG_PATH] [NUM_DEVICES]
if [ $# -lt 2 ]; then
    echo "Usage: bash run_standalone_train_gpu.sh [CONFIG_PATH] [NUM_DEVICES]
    parameters is saved in yaml config file, ."
exit 1
fi

export PYTHONPATH=$PWD
echo "export PYTHONPATH=$PWD"

CONFIG_PATH=$1
NUM_DEVICES=$2
user=$(env | grep USER | cut -d "=" -f 2)
if [ $user == "root" ]; 
then
    echo "Run as root"
    mpirun -n ${NUM_DEVICES} --allow-run-as-root python train.py --config ${CONFIG_PATH} > train.log 2>&1 &
else
    echo "Run as $user"
    mpirun -n ${NUM_DEVICES} python train.py --config ${CONFIG_PATH} > train.log 2>&1 &
fi