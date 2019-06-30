#! /bin/bash
# $1 is the configs folder path
# $2 is the command folder path
set -x
configs_folder=$1
commands_folder=$2
configs=`find ${configs_folder} -name "*.sh"`
commands=`find ${commands_folder} -name "*.sh"`
ARG_COMMENT="#gama for focal loss cross entropy"
arg_name_lc="focal_loss_gama"
ARG_NAME="FOCAL_LOSS_GAMA"
ARG_VALUE="32"

for i in $configs; do
  if grep -Fq "${ARG_NAME}" $i; then
    sed -i "/^${ARG_COMMENT}/d" $i
    sed -i "/^${ARG_NAME}/d" $i
  else
    printf "args in $i already has already removed"
  fi
done

# declare -a scripts=( 'prepare.sh' 'train.sh' 'dev.sh' 'train_dev.sh' )

for j in ${commands[@]}; do
  if grep -Fq "${arg_name_lc}" $j; then
    sed -i "/--${arg_name_lc}/d" $j
  else
    printf "args in $j already removed"
  fi
done

set +x
