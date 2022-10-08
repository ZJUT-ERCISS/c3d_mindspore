#!/bin/bash
# bash run_standalone_train_gpu.sh [CONFIG_PATH]
if [ $# -lt 1 ]; then
    echo "Usage: bash run_standalone_train_gpu.sh [CONFIG_PATH]
    parameters is saved in yaml config file."
exit 1
fi

export PYTHONPATH=$PWD
echo "export PYTHONPATH=$PWD"
CONFIG_PATH=$1
python eval.py --config ${CONFIG_PATH} > eval.log 2>&1 &