#!/usr/bin/env python3
"""
Prints out all the arguments provided (except argument 0, which is the executable name)
"""

import os
import sys
    
rank = os.environ.get('RANK', '0')
prefix = f"{rank:>3}\t"

print(f"{prefix}\tArguments = {' '.join(sys.argv[1:])}")
