#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh

## function template for tokenize dataset.
## $1 is the tokenized_file
## $2 is the output_file
preprocess_file(){
  if [ ! -f $2 ]; then
    echo "preprocessing data for task1, $1"
    python $ROOT_DIR/tensorflow/feature_utils.py --token_case_dict=${TOKEN_CASE_FILE} --tokenized_file=$1 > $2
  else
    echo "WARNING!!! $2 already existed, skipping preprocess_file"
  fi
}

pushd ${RO_DATA_DIR}
data_folders=`find . -mindepth 1 -maxdepth 1 -type d -name "psyc_MISC*" -printf '%f\n'`
for i in $data_folders; do
    input_folder=$RO_DATA_DIR/$i
    output_folder=$DATA_DIR/$i
    if [ ! -f $output_folder ]; then
        mkdir -p $output_folder
        pushd $input_folder
        raw_jsons=`ls *.json`
        popd
        for f in $raw_jsons; do
            # data preparation (vocab extraction from train data)
            preprocess_file $input_folder/${f} $output_folder/${f}.tokenized
        done
    else
        echo "$output_folder already existed"  
    fi
done
popd
