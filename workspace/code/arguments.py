#!/usr/bin/env python3
import os
import sys
    
rank = os.environ.get('RANK', '0')
prefix = f"{rank:>3}\t"

print(f"{prefix}\tArguments = {' '.join(sys.argv[1:])}")
