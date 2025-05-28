#!/bin/bash

OMP_NUM_THREADS=1 \
torchrun --nnodes 1 \
         --nproc_per_node 4 \
         --rdzv_id $RANDOM \
         --rdzv_backend c10d \
         --rdzv_endpoint localhost:29500 \
         $@
