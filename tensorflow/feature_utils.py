# -*- coding: utf-8 -*

# Time-stamp: <2019-06-04 14:49:34 jcao>
# --------------------------------------------------------------------
# File Name          : feature_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A warp of tools for tokenization, flatten, indent json
# --------------------------------------------------------------------

import sys
import argparse
from classes.tokenizer import tokenizer
from classes.FeatureAnnotator import FeatureAnnotator

reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenized_file', nargs='?', required=False, help='file to annotate with tokenized text')
    parser.add_argument('--ext_knowledge_file', nargs='?', required=False, help='file to annotate with tokenized text')
    parser.add_argument('--flatten_file', nargs='?', required=False, help='file to make it flatten, one example per line')
    parser.add_argument('--indent_file', nargs='?', required=False, help='file to make it indent, pretty printed for each example')
    parser.add_argument('--token_case_dict', nargs='?', required=False, help='file to load or store the tokenization special cases')

    args = parser.parse_args()
    if args.tokenized_file is not None:
        # tokenize json file
        tok = tokenizer(token_case_dict=args.token_case_dict, ext_knowledge_file=args.ext_knowledge_file)
        tok.tokenizeDialogueDataset(args.tokenized_file)

    if args.flatten_file is not None:
        FeatureAnnotator.flatten_file(args.flatten_file)

    if args.indent_file is not None:
        FeatureAnnotator.indent_file(args.indent_file)
