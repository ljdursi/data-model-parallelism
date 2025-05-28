#!/bin/bash

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
         --nproc_per_node 4 \
         --rdzv_id ${id} \
         --rdzv_backend c10d \
         --rdzv_endpoint ${rendezvous_server}:29500 \
         $@
