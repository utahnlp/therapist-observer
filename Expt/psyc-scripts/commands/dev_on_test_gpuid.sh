#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# import env variables
. ./env.sh

GPUID_STR=$3

if [[ "$1" = /* ]]; then
  config_data=$1
else
  config_data=$DIR/$1
fi

### configurate data directory
if [ ! -f ${config_data} ]; then
  echo "${config_data} doesn't exist"
  exit $?
else
  . ${config_data}
  echo "run ${config_data}"
fi

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

mkdir -p ${TRAINING_DIR}/results_on_test

pargs="
--use_batch_norm=${USE_BATCH_NORM} \
--use_r_in_wm_seq=${USE_R_IN_WM_SEQ} \
--t_model_config=${T_MODEL_CONFIG} \
--p_model_config=${P_MODEL_CONFIG} \
--decode_combine=${DECODE_COMBINE} \
--use_r_in_seq=${USE_R_IN_SEQ} \
--focal_loss_gama=${FOCAL_LOSS_GAMA} \
--loss_weights=${LOSS_WEIGHTS} \
--gpu=${GPUID_STR} \
--snt_rnn_type=${SNT_RNN_TYPE} \
--snt_seq_hops=${SNT_SEQ_HOPS} \
--snt_att_algo=${SNT_ATT_ALGO} \
--num_att_head=${NUM_ATT_HEAD} \
--dropout_keep_prob_mlp=${DROPOUT_KEEP_PROB_MLP} \
--train_label_embedding=${TRAIN_LABEL_EMBEDDING} \
--train_speaker_embedding=${TRAIN_SPEAKER_EMBEDDING} \
--use_response_s=${USE_RESPONSE_S} \
--decode_r_with_flatten_pu_labels=${DECODE_R_WITH_FLATTEN_PU_LABELS} \
--cluster_strategy=${CLUSTER_STRATEGY} \
--filter_sizes=${FILTER_SIZES} \
--speaker_embed_dim=${SPEAKER_EMBED_DIM} \
--label_embed_dim=${LABEL_EMBED_DIM} \
--use_label_embedding=${USE_LABEL_EMBEDDING} \
--use_speaker_embedding=${USE_SPEAKER_EMBEDDING} \
--decode_goal=${DECODE_GOAL} \
--use_response_u=${USE_RESPONSE_U} \
--response_hops=${RESPONSE_HOPS} \
--decode_r=${DECODE_R} \
--hierarchy_r_pu_att=${HIERARCHY_R_PU_ATT} \
--flat_r_c_att=${FLAT_R_C_ATT} \
--flat_c_r_att=${FLAT_C_R_ATT} \
--decode_func=${DECODE_FUC} \
--use_concat_p=${USE_CONCAT_P} \
--decode_inputs=${DECODE_INPUTS} \
--elmo_emb_output=${ELMO_EMB_OUTPUT} \
--elmo_vocab_file=${ELMO_VOCAB_FILE} \
--elmo_option_file=${ELMO_OPTION_FILE} \
--elmo_weight_file=${ELMO_WEIGHT_FILE} \
--elmo_positions=${ELMO_POSITIONS} \
--elmo_snt_dict_file=${ELMO_SNT_DICT_FILE} \
--elmo_u_cache_file=${ELMO_U_CACHE_FILE} \
--elmo_p_cache_file=${ELMO_P_CACHE_FILE} \
--elmo_q_cache_file=${ELMO_Q_CACHE_FILE} \
--use_character_elmo=${USE_CHARACTER_ELMO} \
--model_prefix_to_restore=${ALGO}$2 \
--test_files=${TEST_FILES} \
--dev_files=${TEST_FILES} \
--train_files=${TRAIN_FILES} \
--use_shared_encoding=${USE_SHARED_ENCODING} \
--rnn_type=${RNN_TYPE} \
--loss_func=${LOSS_FUNC} \
--ema_decay=${EMA_DECAY} \
--use_hierarchy_selfatt=${USE_HIERARCHY_SELFATT} \
--hierarchy_q_pu_att=${HIERARCHY_Q_PU_ATT} \
--max_u_len=${MAX_U_LEN} \
--dial_encode=${DIAL_ENCODE} \
--dropout_keep_prob_emb=${DROPOUT_KEEP_PROB_EMB} \
--flat_q_c_att=${FLAT_Q_C_ATT} \
--char_embed_size=${CHAR_EMBED_SIZE} \
--word_embed_size=${WORD_EMBED_SIZE} \
--topM_for_acc_prob=${TOPM_FOR_ACC_PROB} \
--topK_list=${TOPK_LIST} \
--decode_q=${DECODE_Q} \
--decode_p=${DECODE_P} \
--char_emb_file=${CHAR_EMB_FILE} \
--word_emb_file=${WORD_EMB_FILE} \
--token_key_to_use=${TOKEN_KEY_TO_USE} \
--pos_weight=${POS_WEIGHT} \
--flat_c_q_att=${FLAT_C_Q_ATT} \
--max_p_len=${MAX_P_LEN} \
--max_q_len=${MAX_Q_LEN} \
--context_window=${CONTEXT_WINDOW} \
--question_window=${QUESTION_WINDOW} \
--acc_sum_prob=${ACC_SUM_PROB} \
--max_num_char_to_keep_forward=${MAX_NUM_CHAR_TO_KEEP_FORWARD} \
--max_num_char_to_keep_backward=${MAX_NUM_CHAR_TO_KEEP_BACKWARD} \
--num_filters=${NUM_FILTERS} \
--evaluate \
--algo=${ALGO} \
--train_embedding=${TRAIN_EMBEDDING} \
--use_char_embedding=${USE_CHAR_EMBED} \
--use_selfatt=${USE_SELFATT} \
--hidden_size=${HIDDEN_SIZE} \
--learning_rate=${LEARNING_RATE} \
--weight_decay=${WEIGHT_DECAY} \
--batch_size=${BATCH_SIZE} \
--vocab_dir=${VOCAB_DIR} \
--model_dir=$TRAINING_DIR/models/ \
--result_dir=$TRAINING_DIR/results_on_test/ \
--summary_dir=$TRAINING_DIR/summary/ \
"

python $ROOT_DIR/tensorflow/dial_run.py $pargs &> $TRAINING_DIR/dev_on_test_${ALGO}$2.log
