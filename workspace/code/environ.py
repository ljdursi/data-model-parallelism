#!/usr/bin/env python3
"""
Prints all the environment variables that contain a keyword
"""
import os

keywords = ["WORLD", "RANK", "PORT", "MASTER", "LOCAL"]

prefix = ''
if 'RANK' in os.environ.keys():
    rank = int(os.environ['RANK'])
    prefix = f"{rank:3d}\t"

for k, v in os.environ.items():
    output = False
    for word in keywords:
        if word in k:
            output = True
            break
    if output:
        print(f"{prefix}{k:25s}\t\t{v}")
