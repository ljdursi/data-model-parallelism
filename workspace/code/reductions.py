#!/usr/bin/env python3
"""
Demonstrate data reductions across ranks
"""
import argparse
import os
import time

import torch
import torch.distributed as dist

global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

parser = argparse.ArgumentParser(description='Synchronization example')

args = parser.parse_args()

# create group of processes which will synchronize
# gloo is a simple default backend for CPUs, esp single-node
device = torch.device("cpu")
dist.init_process_group(backend="gloo", world_size=world_size, 
                        rank=global_rank)

# e.g. on rank 0 with world_size 4, input will be [0, 0, 0, 0]
local_result = torch.tensor([global_rank]*world_size) 
print(f"{global_rank}: local value = {local_result}", flush=True)

# sum across all ranks
to_be_summed = torch.tensor([global_rank]*world_size) 
dist.all_reduce(to_be_summed, op=dist.ReduceOp.SUM)
print(f"{global_rank}: summed value = {to_be_summed}", flush=True)

# all_gather
to_be_gathered = torch.tensor([global_rank]) 
result = [torch.tensor([0]) for _ in range(world_size)]
dist.all_gather(result, to_be_gathered)
print(f"{global_rank}: allgathered value = {result}", flush=True)


# get rid of the process group
dist.destroy_process_group()
