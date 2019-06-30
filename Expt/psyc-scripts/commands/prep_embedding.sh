#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
# the data dir contains those corpus
RO_DATA_DIR=$ROOT_DIR/Expt/data/psyc_ro/

## function template for tokenize dataset.
## $1 is the tokenized_file
## $2 is the output_file
tokenize_snt_file(){
  if [ ! -f $2 ]; then
    echo "tokenized for snt file $1"
    python $ROOT_DIR/tensorflow/classes/embedding_utils.py --input $1 --tokenized_file=$2 
  else
    echo "WARNING!!! $2 already existed, skipping preprocess_file"
  fi
}

prepare_elmo_file(){
  if [ ! -f $2 ]; then
    echo "prepare vocab for elmo file $1"
    python $ROOT_DIR/tensorflow/classes/embedding_utils.py --input $1 --elmo_vocab=$2 
  else
    echo "WARNING!!! $2 already existed, skipping generate elmo vocab"
  fi
}

tokenize_snt_file $1 $1.tokenized
prepare_elmo_file $1.tokenized $1.elmo_vocab
