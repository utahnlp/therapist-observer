#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
# the data dir contains those corpus
RO_DATA_DIR=$ROOT_DIR/Expt/data/psyc_ro/
RO_DOWNLOAD_DATA_DIR=$ROOT_DIR/Expt/data/psyc_ro/download/

DATA_DIR=$ROOT_DIR/Expt/data/psyc/
TOKEN_CASE_FILENAME=psyc.token-case.json
TOKEN_CASE_FILE=$DATA_DIR/$TOKEN_CASE_FILENAME

## function template for place and create sanity check data
## $1 is the input folder name
place_data(){
  output_dir=$RO_DATA_DIR/$1
  echo "starting to place_data for $1"
  if [ ! -f $output_dir ]; then
    mkdir -p $output_dir
    cp ./$1/train.json $output_dir
    cp ./$1/dev.json $output_dir
    cp ./$1/test.json $output_dir
    shuf $output_dir/train.json | head -n 20 > $output_dir/train_sc20.json
    shuf $output_dir/train_sc20.json | head -n 10 > $output_dir/dev_sc10.json
  else
    echo "WARNING!!! $1 already existed, please remove it !"
  fi
}

pushd $RO_DOWNLOAD_DATA_DIR/ori_splits/
data_folders=`find . -mindepth 1 -maxdepth 1 -type d -name "psyc_MISC*" -printf '%f\n'`
echo "$data_folders"
for i in $data_folders; do
    # place data from every data_folder
    place_data $i
done
popd
