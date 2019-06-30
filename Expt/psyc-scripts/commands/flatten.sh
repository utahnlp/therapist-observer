#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh

if [[ "$1" = /* ]]; then
  config_data=$1
else
  config_data=$DIR/$1
fi

### configurate data directory
if [ ! -f ${config_data} ]; then
  echo "${config_data} doesn't exist"
  exit $?
else
  . ${config_data}
  echo "run ${config_data}"
fi

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

flatten_file(){
  if [ ! -f $2 ]; then
    echo "make $1 flatten"
    python $ROOT_DIR/tensorflow/feature_utils.py --flatten_file=$1 > $2
  else
    echo "$2 already existed, skipping preprocess_dataset"
  fi
}

# data preparation (vocab extraction from train data)
flatten_file $2 $3
