#!/bin/bash
set -x
PIPE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. ./env.sh

if [[ "$1" = /* ]]; then
  config_data=$1
else
  config_data=${PIPE_DIR}/$1
fi


# train model
$PIPE_DIR/train.sh ${config_data}

# evaluate model and write out prediction on most related paragraph per document, and return loss score
$PIPE_DIR/eval_test_all.sh ${config_data}

# similar to dev.sh but this applies to test.json without annotation and cannot return loss score
#$PIPE_DIR/pred.sh ${config_data}

# tar czvf dev
#tar czvf ${WORK_DIR} ${WORK_DIR}.dev

set +x
