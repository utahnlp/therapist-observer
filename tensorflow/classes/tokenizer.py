# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 15:08:55 jcao>
# --------------------------------------------------------------------
# File Name          : tokenizer.py
# Original Author    : jiessie.cao@gmail.com
# Description        : tokenization with spacy
# --------------------------------------------------------------------

import sys
import os
import ujson as json
import re
from tqdm import tqdm
from spacy.lang.en import English
from spacy.attrs import ORTH, IS_PUNCT
from spacy.matcher import Matcher
reload(sys)
sys.setdefaultencoding('utf8')


class tokenizer(object):
    def __init__(self, token_case_dict, ext_knowledge_file=None):
        self.annotator = English()
        self.tokenizer = self.annotator.tokenizer

    #################################
    #### TOKENIZATION FUNCTIONS #####
    #################################

    # convert a string into a list of tokens
    # this function is left here for compactibility issue
    def tokenized_str_array(self, text):
        return [tok.text for tok in self.annotator(text)]

    def tokenize_str(self, text):
        return [{'token': tok.text.lower(), 'type': 'normal_token', 'is_stop': tok.is_stop} for tok in self.annotator(text)]

    def _make_normal_token(self, token, is_stop=False):
        return {'token': token.lower(), 'type': 'normal_token', 'is_stop': is_stop}

    # this function takes in a text and does tokenization on text
    def tokenize(self, text):
        tokens = []
        doc = self.annotator(text)
        i = 0
        while i < len(doc):
            tokens.append(self._make_normal_token(doc[i].text, doc[i].is_stop))
            i += 1
        return tokens

    # tokenize one sample from the dialogue dataset
    def tokenizeDialogueSample(self, dialogueSample):
        for utterance in dialogueSample['messages-so-far']:
            text = utterance['utterance']
            utterance['tokenized_utterance'] = self.tokenize(text)

        for correct_answer in dialogueSample['options-for-correct-answers']:
            text = correct_answer['utterance']
            correct_answer['tokenized_utterance'] = self.tokenize(text)

    # tokenize a dialogue dataset line by line
    def tokenizeDialogueDataset(self, json_file):
        if os.path.isfile(json_file) and os.path.exists(json_file):
            with open(json_file, "r") as ro_raw_file:
                for line in tqdm(ro_raw_file):
                    sample = json.loads(line)
                    self.tokenizeDialogueSample(sample)
                    #try:
                    #    self.tokenizeDialogueSample(sample)
                    #except Exception as err:
                    #    raise RuntimeError("sample {} cannot be tokenized, error is {}".format(sample['example-id'], err))
                    # print sample line by line
                    print(json.dumps(sample, ensure_ascii=False))
        else:
            raise IOError('Invalid  path {}'.format(json_file))
