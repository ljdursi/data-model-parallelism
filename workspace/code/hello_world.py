#!/usr/bin/env python3
"""
Prints hello world from each rank
"""
import os

try:
    global_rank = os.environ["RANK"]
    local_rank = os.environ["LOCAL_RANK"]
    world_size = os.environ["WORLD_SIZE"]

    print(f"Hello, world from rank {global_rank} of {world_size}! (local rank {local_rank})")
except:
    print(f"Hello, world!")

