#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
# the data dir contains those corpus
RO_DATA_DIR=$ROOT_DIR/Expt/data/psyc_ro/
DATA_DIR=$ROOT_DIR/Expt/data/psyc/
TOKEN_CASE_FILENAME=psyc.token-case.json
TOKEN_CASE_FILE=$DATA_DIR/$TOKEN_CASE_FILENAME

#declare -a remove_stops=( 0 1 )
declare -a remove_stops=( 0 )
declare -a replace_tokens=( "no" )


pushd ${DATA_DIR}
data_folders=`find . -mindepth 1 -maxdepth 1 -type d -name "psyc_MISC*" -printf '%f\n'`
for i in $data_folders; do
  input_folder=$DATA_DIR/$i
  for stop in ${remove_stops[@]}; do
    for replace in ${replace_tokens[@]}; do
      output_folder=${DATA_DIR}/prep_data/rmstop_${stop}_rpt_${replace}/${i}/
      echo "tokenizing with stop=$stop, replace=$replace"
      python $ROOT_DIR/tensorflow/preprocess.py --input_json_files ${input_folder}/*.tokenized --output_folder ${output_folder} --remove_stops $stop --replace_tokens $replace
    done
  done
done
popd

# remove stops = 0, not removing
# remove stops = 1, remove stop words
# replace_tokens = no, title
