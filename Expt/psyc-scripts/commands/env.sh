#!/bin/bash
# get current folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
# root dir is the root folder of the cloned code folder, you can specify an
# absolute folder as well, in case you have change the folder strucutre.
ROOT_DIR=$DIR/../../../
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
# echo "root_dir is: "$ROOT_DIR

# prepare python environment
eval "$(pyenv init -)"
pyenv activate py2.7_tf1.4

# META_DATA_CONFIG, include dev_id, tokenization rules
META_DATA_CONFIG=$ROOT_DIR/Expt/data/psyc_ro/meta_data_config/

# data dir contains those corpus, RO_DATA_DIR is the readonly data
RO_DATA_DIR=$ROOT_DIR/Expt/data/psyc_ro/
# DATA_DIR is for data building and training
DATA_DIR=$ROOT_DIR/Expt/data/psyc/
# used by spacy , the file name, if no specific token-case, please just ignore this.
TOKEN_CASE_FILENAME=psyc.token-case.json
# used by spacy , the file name, if no specific token-case, please just ignore this.
TOKEN_CASE_FILE=$META_DATA_CONFIG/$TOKEN_CASE_FILENAME
