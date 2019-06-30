#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh
MISC_CODE=$1
WE=$2
WINDOW=$3

## function template for tokenize dataset.
## $1 is the converstion name for MISC code
## $2 is whether drop_exception code
## $3 is the original psyc file
transform_file(){
  output_file=$RO_DATA_DIR/download/all_dial_$1.json
  if [ ! -f $output_file ]; then
    echo "transform original data to json format with $1, $output_file"
    python $ROOT_DIR/tensorflow/psyc_utils.py --transform_from $3 --cluster_strategy $1_$2 --full_dial $output_file
  else
    echo "WARNING!!! $output_file already existed, skipping transforming original psyc file"
  fi
}

#transform_file $1 $RO_DATA_DIR/download/miscall_all.txt
transform_file $MISC_CODE $WE $RO_DATA_DIR/download/miscall_all_fix.txt

seg_file(){
  output_folder=$RO_DATA_DIR/download/ori_splits/psyc_${1}_${3}_padding
  if [ ! -d $output_folder ]; then
    mkdir -p $output_folder
    echo "Seg full dialogue $2 into windows with cluster_strategy=$1, window_size=$3, into $output_folder"
    python $ROOT_DIR/tensorflow/psyc_utils.py --padding_ahead --cluster_strategy $1 --full_dial $2 --seg_dial_by_utterance $3 --output_folder $output_folder
  else
    echo "WARNING!!! $output_folder already existed, skipping transforming original psyc file"
  fi
}

seg_file ${MISC_CODE}_${WE} $RO_DATA_DIR/download/all_dial_$1.json ${WINDOW}

