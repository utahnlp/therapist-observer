#!/bin/bash
# import global variables
. ./env.sh

dev_id_file=$META_DATA_CONFIG/dev_id

./trans_psyc_padding_w_devids.sh MISC28 ML 17 $dev_id_file
#./trans_psyc_padding_w_devids.sh MISC28 EL 17 $dev_id_file
#./trans_psyc_padding_w_devids.sh MISC28 WOE 17 $dev_id_file
