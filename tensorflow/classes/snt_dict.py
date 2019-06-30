# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 15:08:17 jcao>
# --------------------------------------------------------------------
# File Name          : snt_dict.py
# Original Author    : jiessie.cao@gmail.com
# Description        :
# This module implements the SentenceDict class for converting sentence to id and back
# --------------------------------------------------------------------

import numpy as np
import h5py
import os


class SentenceDict(object):
    """
    Implements a dict to store the sentence in the data, with their corresponding embeddings.
    """
    def __init__(self, filename=None, lower=False, init_snts=None):
        self.id2snt = []
        self.snt2id = {}
        self.snt_cnt = {}
        self.lower = lower
        self.filename = filename
        if init_snts:
            for i in init_snts:
                self.add(i)

        self.embeddings = None

        if filename is not None and os.path.exists(filename):
            self.load_from_file(filename)

    def size(self):
        """
        get the size of sntulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2snt)

    def load_from_file(self, file_path):
        """
        loads the snt from file_path
        Args:
            file_path: a file with a word in each line
        """
        for line in open(file_path, 'r'):
            snt = line.rstrip('\n')
            self.add(snt)

    def get_id(self, snt):
        """
        gets the id of a snt, returns the id of unk snt if snt is not in snt
        Args:
            key: a string indicating the word
        Returns:
            an integer
        throw KeyError exception, if no that snt
        """
        snt = snt.lower() if self.lower else snt
        return self.snt2id[snt]

    def get_snt(self, idx):
        """
        gets the snt corresponding to idx, returns unk snt if idx is not in snt
        Args:
            idx: an integer
        returns:
            a snt string
        throw KeyError exception, if no that idx
        """
        return self.id2snt[idx]

    def add(self, snt, cnt=1):
        """
        adds the snt to snt
        Args:
            snt: a string
            cnt: a num indicating the count of the snt to add, default is 1
        """
        snt = snt.lower() if self.lower else snt
        if snt in self.snt2id:
            idx = self.snt2id[snt]
        else:
            idx = len(self.id2snt)
            self.id2snt.append(snt)
            self.snt2id[snt] = idx
        if cnt > 0:
            if snt in self.snt_cnt:
                self.snt_cnt[snt] += cnt
            else:
                self.snt_cnt[snt] = cnt
        return idx

    def load_pretrained_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        snts not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        embedding_paths = [embedding_path]

        for embedding_path in embedding_paths:
            with h5py.File(embedding_path, 'r') as fin:
                self.embeddings = fin

    def convert_to_ids(self, snts):
        """
        Convert a list of snts to ids, use unk_snt if the snt is not in snt.
        Args:
            snts: a list of snt
        Returns:
            a list of ids
        """
        vec = [self.get_id(snt) for snt in snts]
        return vec

    def dump_snts_file_without_start_end(self):
        """
        Dump snts without adding sos and eos
        """
        with open('{}.without'.format(self.filename), 'w') as fout:
            for snt in self.id2snt:
                fout.write("{}\n".format(snt))

    def dump_snts_file_with_start_end(self):
        """
        Dump snts by adding sos and eos
        """
        with open('{}.with'.format(self.filename), 'w') as fout:
            for snt in self.id2snt:
                if not snt:
                    fout.write("{}\n".format('<S> </S>'.format(snt)))
                else:
                    fout.write("{}\n".format('<S> {} </S>'.format(snt)))
