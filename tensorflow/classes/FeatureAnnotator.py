# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 14:54:13 jcao>
# --------------------------------------------------------------------
# File Name          : FeatureAnnotator.py
# Original Author    : jiessie.cao@gmail.com
# Description        :
# --------------------------------------------------------------------

import sys
import os
import ujson as json
import re

reload(sys)
sys.setdefaultencoding('utf8')


class FeatureAnnotator(object):
    @staticmethod
    def flatten_file(indented_file):
        """
        make a file flatten into indented file
        """
        if os.path.isfile(indented_file) and os.path.exists(indented_file):
            with open(indented_file, "r") as fin:
                examples = json.load(fin)
                for example in examples:
                    print(json.dumps(example, ensure_ascii=False))

    @staticmethod
    def indent_file(flatten_file):
        """
        make an indented file into a flatten file
        """
        if os.path.isfile(flatten_file) and os.path.exists(flatten_file):
            with open(flatten_file, "r") as fin:
                samples = []
                for lidx, line in enumerate(fin):
                    sample = json.loads(line.strip())
                    samples.append(sample)
                print(json.dumps(samples, indent=4, ensure_ascii=False))
