#!/bin/bash
set -x
PIPE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh
# download dataset and glove
# echo "Downloading dataset and glove embedding..."
# $PIPE_DIR/download.sh || { echo 'Downloading failed' ; exit 1 ; }

# transform fron the csv to json format
$PIPE_DIR/trans.sh || { echo 'trans failed' ; exit 1 ;}

# echo "place data into specific folder for later preprocessing"
$PIPE_DIR/place_data.sh || { echo 'Place data failed' ; exit 1 ; }

#tokenizing data
echo "Tokenizing data..."
$PIPE_DIR/tok.sh || { echo 'Tokenize dataset failed' ; exit 1 ; }

# prepare vocab (can pretrained embedding)
echo "Preprocessing data..."
$PIPE_DIR/preprocess_dataset.sh || { echo 'Preprocessing dataset failed' ; exit 1 ; }

# shuf and split those preprocessed datafile
# $PIPE_DIR/shufsplit_dataset.sh || { echo 'Shuf and split dataset failed' ; exit 1 ; }

set +x
