#!/bin/bash
set -x
PIPE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. ./env.sh

if [[ "$1" = /* ]]; then
  config_data=$1
else
  config_data=${PIPE_DIR}/$1
fi

#$PIPE_DIR/dev_on_test.sh ${config_data}
$PIPE_DIR/dev_on_test.sh ${config_data} _F1_macro
# $PIPE_DIR/dev.sh ${config_data} _F1_macro
#$PIPE_DIR/dev_on_test.sh ${config_data} _F1_weighted_macro
#$PIPE_DIR/dev_on_test.sh ${config_data} _F1_micro

set +x
