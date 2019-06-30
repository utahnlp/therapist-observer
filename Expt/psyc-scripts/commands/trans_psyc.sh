#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
# the data dir contains those corpus
RO_DATA_DIR=$ROOT_DIR/Expt/data/psyc_ro/
DATA_DIR=$ROOT_DIR/Expt/data/psyc_ro/



## function template for tokenize dataset.
## $1 is the original psyc file
## $2 is the output_file
transform_file(){
  if [ ! -f $2 ]; then
    echo "transform original data to json format, $1"
    python $ROOT_DIR/tensorflow/psyc_utils.py --transform_from $1 --full_dial $2
  else
    echo "WARNING!!! $2 already existed, skipping transforming original psyc file"
  fi
}

transform_file $RO_DATA_DIR/download/miscall_all.txt $RO_DATA_DIR/download/miscall_all_dial.json

seg_file(){
  output_folder=$RO_DATA_DIR/download/ori_splits/$2
  if [ ! -d $output_folder ]; then
    mkdir -p $output_folder
    echo "Seg full dialogue into windows, $1 into $output_folder"
    python $ROOT_DIR/tensorflow/psyc_utils.py --full_dial $1 --seg_dial_by_utterance $2 --output_folder $output_folder
  else
    echo "WARNING!!! $output_folder already existed, skipping transforming original psyc file"
  fi
}

seg_file $RO_DATA_DIR/download/miscall_all_dial.json $1

