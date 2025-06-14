#!/bin/bash

if [ $# -lt 4 ]
then
    echo >&2 "ERROR: not enough arguments provided"
    echo >&2 ""
    echo >&2 "Usage: $0 number-of-processes-per-node min-nodes max-nodes script-to-run [arg1 arg2...]"
    echo >&2 "       e.g. $0 4 3 4 ./arguments.py a b c 1233"
    exit 1
else
    nproc_per_node=$1
    min_nnodes=$2
    max_nnodes=$3
    shift 
fi

# this identifies the run. It can be anything,
# but if you're running torchrun across multiple nodes,
# it has to be the same for each of them

id=$RANDOM

# identifies one of the torchrun servers as the one
# to use for coordination.
# Here there's just one node, so we're using localhost
# but if you're running torchrun across multiple nodes,
# all the nodes have to agree; often the first node 
# (in, e.g., SLURM_JOB_NODELIST) is chosen

rendezvous_server=localhost

NTHREADS=4 # should be set up to be number of CPU cores / number of processes, generally, but for this tutorial we set the number low

OMP_NUM_THREADS=${NTHREADS} \
torchrun --nnodes ${min_nnodes}:${max_nnodes} \
         --max-restarts=3 \
         --nproc_per_node ${nproc_per_node} \
         --rdzv_id ${id} \
         --rdzv_backend c10d \
         --rdzv_endpoint ${rendezvous_server}:29500 \
         $@
