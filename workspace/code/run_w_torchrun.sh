#!/bin/bash

if [ $# -lt 2 ]
then
    echo >&2 "ERROR: not enough arguments provided"
    echo >&2 ""
    echo >&2 "Usage: $0 number-of-processes script-to-run [arg1 arg2...]"
    echo >&2 "       e.g. $0 4 ./arguments.py a b c 1233"
    exit 1
else
    nproc_per_node=$1
    shift 
fi

NTHREADS=4 # should be set up to be number of CPU cores / number of processes, generally

OMP_NUM_THREADS=${NTHREADS} \
torchrun --standalone \
         --nproc_per_node ${nproc_per_node} \
         $@
