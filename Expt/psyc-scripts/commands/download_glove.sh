#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh

if [ ! -e $DATA_DIR ]; then
  mkdir -p $DATA_DIR
fi

if [ ! -e $RO_DATA_DIR ]; then
  mkdir -p $RO_DATA_DIR
fi

echo "Downloading glove 840B.300d"
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $RO_DATA_DIR/glove.840B.300d.zip
unzip $RO_DATA_DIR/glove.840B.300d.zip -d $RO_DATA_DIR
rm $RO_DATA_DIR/glove.840B.300d.zip
