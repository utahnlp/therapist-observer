#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

indent_file(){
  if [ ! -f $2 ]; then
    echo "make $1 indented"
    python $ROOT_DIR/tensorflow/feature_utils.py --indent_file=$1 > $2
  else
    echo "$2 already existed, skipping preprocess_dataset"
  fi
}

# data preparation (vocab extraction from train data)
indent_file $1 $2
