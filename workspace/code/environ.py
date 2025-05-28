#!/usr/bin/env python3
"""
Prints all the environment variables that contain a keyword
"""
import os

rank = os.environ.get('RANK', '0')
prefix = f"{rank:>3}\t"

keywords = ["WORLD", "RANK", "PORT", "MASTER", "LOCAL"]

for k, v in os.environ.items():
    if any(word in k for word in keywords):
        print(f"{prefix}{k:25s}\t{v}")
