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

OMP_NUM_THREADS=1 \
torchrun --nnodes 1 \
         --nproc_per_node ${nproc_per_node} \
         --rdzv_id ${id} \
         --rdzv_backend c10d \
         --rdzv_endpoint ${rendezvous_server}:29500 \
         $@
