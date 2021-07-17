#!/bin/bash
PYTHON_VERSION="3.8"
if [ "$1" = "" ]; then
    printf "\n ERROR: Please provide an name for the new conda environment\n\n"
else
    if ! [ "$2" = "" ]; then
        PYTHON_VERSION=$2
    fi
    conda init
    conda create -y --name $1 python=$PYTHON_VERSION
    eval "$(conda shell.bash hook)"
    conda activate $1
    pip install -r requirements.txt
fi
