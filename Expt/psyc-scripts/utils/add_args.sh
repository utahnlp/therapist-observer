#! /bin/bash
# $1 is the configs folder path
# $2 is the command folder path
set -x
configs_folder=$1
commands_folder=$2
configs=`find ${configs_folder} -name "*.sh"`
commands=`find ${commands_folder} -name "*.sh"`
ARG_COMMENT="# whether use batch normalization"
arg_name_lc="use_batch_norm"
ARG_NAME="USE_BATCH_NORM"
ARG_VALUE=""

for i in $configs; do
  if grep -Fq "${ARG_COMMENT}" $i; then
    printf "$i already has already taken effect"
  else
    printf "${ARG_COMMENT}\n${ARG_NAME}=${ARG_VALUE}\n" >> $i
  fi
done

# declare -a scripts=( 'prepare.sh' 'train.sh' 'dev.sh' 'train_dev.sh' )

for j in ${commands[@]}; do
  if grep -Fq "${ARG_NAME}" $j; then
    printf "$j already take effect"
  else
    sed -i "/pargs=\"/a --${arg_name_lc}=\$\{${ARG_NAME}\} \\\\" $j
  fi
done

set +x
