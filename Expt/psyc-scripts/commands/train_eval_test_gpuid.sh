#!/bin/bash
set -x
PIPE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. ./env.sh

if [[ "$1" = /* ]]; then
  config_data=$1
else
  config_data=${PIPE_DIR}/$1
fi

GPUID_STR=$2

# train model
$PIPE_DIR/train_gpuid.sh ${config_data} ${GPUID_STR}

# evaluate model and write out prediction on most related paragraph per document, and return loss score
$PIPE_DIR/eval_test_all_gpuid.sh ${config_data} ${GPUID_STR}

# similar to dev.sh but this applies to test.json without annotation and cannot return loss score
#$PIPE_DIR/pred.sh ${config_data}

# tar czvf dev
#tar czvf ${WORK_DIR} ${WORK_DIR}.dev

set +x
