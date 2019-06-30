# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 15:05:02 jcao>
# --------------------------------------------------------------------
# File Name          : feeddict_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : This module is about all kinds of dataset and batch preparing tools.
# --------------------------------------------------------------------

def padding(batch_data, key, length, pad):
    """
    according to the length, padding batch_data[key] with pad
    """
    batch_data[key] = [(ids + [pad] * (length - len(ids)))[: length] for ids in batch_data[key]]
