#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run ablation evaluation for all Path A checkpoints."""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from path_a.scripts.run_path_a_ablation import main

if __name__ == '__main__':
    main()
