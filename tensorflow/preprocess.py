# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 14:53:20 jcao>
# --------------------------------------------------------------------
# File Name          : preprocess.py
# Original Author    : jiessie.cao@gmail.com
# Description        : Utils for preprocessing the dataset
# --------------------------------------------------------------------

import argparse, os, sys
import ujson as json
import warnings
from tqdm import tqdm
from multiprocessing import Pool
from classes.preprocess.preprocess_psyc_dataset import preprocess_psyc_dataset
from classes.preprocess.feature_extractor import feature_extractor

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json_files', nargs='*', required=True, help='input json files')
    parser.add_argument('--token_case_dict', nargs='?', required=False, help='token case dict')
    parser.add_argument('--remove_stops', nargs='?', type=int, default=0, help='remove stopwords?')
    parser.add_argument('--output_folder', nargs='?', required=True, help='output folder for those preprocessed data')
    parser.add_argument('--replace_tokens', choices=['no', 'title', 'title2', 'expanded_title', 'expanded_title2'], default='no', help='methods to replace tokens')
    parser.add_argument('--retrain_pipeline', default=False, help='Retrain model?')


    args = parser.parse_args()
    preprocess_psyc_dataset(
        args.input_json_files,
        args.output_folder,
        args.remove_stops,
        args.replace_tokens
    )
