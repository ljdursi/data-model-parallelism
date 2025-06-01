#!/usr/bin/env python3
"""
Synchronize (or not) on various tasks
"""
import argparse
import os
import time

import torch
import torch.distributed as dist

global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

parser = argparse.ArgumentParser(description='Synchronization example')
parser.add_argument('--barrier', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

# create group of processes which will synchronize
# gloo is a simple default backend for CPUs, esp single-node
dist.init_process_group("gloo", world_size=world_size, 
                        rank=global_rank)

for task in range(2):
    print(f"Task {task} starts on rank {global_rank}/{world_size}", flush=True)

    # task takes different lengths of time on different ranks
    time.sleep(global_rank + 1)

    if args.barrier:
        dist.barrier()

    print(f"Task {task} ends on rank {global_rank}/{world_size}", flush=True)
    time.sleep(0.1)


# get rid of the process group
dist.destroy_process_group()
