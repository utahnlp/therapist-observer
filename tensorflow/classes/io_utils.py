# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 15:05:46 jcao>
# --------------------------------------------------------------------
# File Name          : io_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : Io funtions
# --------------------------------------------------------------------

import os
import mmap
from tqdm import tqdm

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
