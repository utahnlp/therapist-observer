# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 15:03:22 jcao>
# --------------------------------------------------------------------
# File Name          : embedding_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : embedding utils for prepare elmo vocab
# --------------------------------------------------------------------

import sys
import os
import codecs
import argparse
import ujson as json
from elmo_utils import ELMo_Utils
from vocab import Vocab
import re
from tqdm import tqdm
from spacy.lang.en import English
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')

class EmbeddingUtils(object):
    def __init__(self):
        self.annotator = English()
        self.tokenizer = self.annotator.tokenizer

    #################################
    #### TOKENIZATION FUNCTIONS #####
    #################################
    def tokenized_str_array(self, text):
        return [tok.text for tok in self.annotator(text)]


    # given a input snt file, tokenized it with spacy and then write into a tokenized snt file
    def tokenize_snt_file(self, input_snt_file, tokenized_file):
        if os.path.exists(tokenized_file):
            print "{} exists, skipping... ".format(input_snt_file)
        else:
            print "Processing: {}".format(tokenized_file)
            with open(tokenized_file, 'w') as f:
                i = 0
                for l in tqdm(codecs.open(input_snt_file, 'r3', encoding='utf-8')):
                    tokenized_snt = embedding_utils.tokenized_str_array(l)
                    concat_str = ' '.join(tokenized_snt)
                    i = i + 1
                    f.write(concat_str +'\n')
                print "wrote {} tokenized lines".format(i)

    # given a tokenized snt file as input, it output a snt file with unique sentence and also with it vocab sorted by frequency
    def prepare_elmo_vocab(self, tokenized_file, elmo_vocab):
        vocab = Vocab(lower=True)
        for l in tqdm(codecs.open(tokenized_file, 'r3', encoding='utf-8')):
            for word in l.split():
                vocab.add(word)
        # write down vocab file for training elmo
        # no need to filter out, because it is use the character level emebdding
        ELMo_Utils.prepare_elmo_vocab_file(vocab, elmo_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input corpus files in line-by-line format')
    parser.add_argument('--tokenized_file', type=str, required=False, help='output tokenized_file of the input')
    parser.add_argument('--elmo_vocab', type=str, required=False, help='the dir to generate elmo prepare files')

    args = parser.parse_args()
    embedding_utils = EmbeddingUtils()
    if args.tokenized_file:
        embedding_utils.tokenize_snt_file(args.input, args.tokenized_file)

    if args.elmo_vocab:
        embedding_utils.prepare_elmo_vocab(args.input, args.elmo_vocab)
