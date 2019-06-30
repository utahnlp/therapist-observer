# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 14:55:39 jcao>
# --------------------------------------------------------------------
# File Name          : config_reader.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A class to read and parse the configurations
# --------------------------------------------------------------------

import subprocess
import argparse

pargs='''
--decode_combine=${DECODE_COMBINE}
--use_r_in_seq=${USE_R_IN_SEQ}
--use_r_in_wm_seq=${USE_R_IN_WM_SEQ}
--use_batch_norm=${USE_BATCH_NORM}
--focal_loss_gama=${FOCAL_LOSS_GAMA}
--loss_weights=${LOSS_WEIGHTS}
--snt_rnn_type=${SNT_RNN_TYPE}
--snt_seq_hops=${SNT_SEQ_HOPS}
--snt_att_algo=${SNT_ATT_ALGO}
--num_att_head=${NUM_ATT_HEAD}
--dropout_keep_prob_mlp=${DROPOUT_KEEP_PROB_MLP}
--train_label_embedding=${TRAIN_LABEL_EMBEDDING}
--train_speaker_embedding=${TRAIN_SPEAKER_EMBEDDING}
--use_response_s=${USE_RESPONSE_S}
--decode_r_with_flatten_pu_labels=${DECODE_R_WITH_FLATTEN_PU_LABELS}
--cluster_strategy=${CLUSTER_STRATEGY}
--filter_sizes=${FILTER_SIZES}
--speaker_embed_dim=${SPEAKER_EMBED_DIM}
--label_embed_dim=${LABEL_EMBED_DIM}
--use_label_embedding=${USE_LABEL_EMBEDDING}
--use_speaker_embedding=${USE_SPEAKER_EMBEDDING}
--decode_goal=${DECODE_GOAL}
--use_response_u=${USE_RESPONSE_U}
--response_hops=${RESPONSE_HOPS}
--decode_r=${DECODE_R}
--hierarchy_r_pu_att=${HIERARCHY_R_PU_ATT}
--flat_r_c_att=${FLAT_R_C_ATT}
--flat_c_r_att=${FLAT_C_R_ATT}
--decode_func=${DECODE_FUC}
--use_concat_p=${USE_CONCAT_P}
--decode_inputs=${DECODE_INPUTS}
--elmo_emb_output=${ELMO_EMB_OUTPUT}
--elmo_vocab_file=${ELMO_VOCAB_FILE}
--elmo_option_file=${ELMO_OPTION_FILE}
--elmo_weight_file=${ELMO_WEIGHT_FILE}
--elmo_positions=${ELMO_POSITIONS}
--elmo_snt_dict_file=${ELMO_SNT_DICT_FILE}
--elmo_u_cache_file=${ELMO_U_CACHE_FILE}
--elmo_p_cache_file=${ELMO_P_CACHE_FILE}
--elmo_q_cache_file=${ELMO_Q_CACHE_FILE}
--use_character_elmo=${USE_CHARACTER_ELMO}
--model_prefix_to_restore=${ALGO}$2
--test_files=${TEST_FILES}
--dev_files=${TEST_FILES}
--train_files=${TRAIN_FILES}
--use_shared_encoding=${USE_SHARED_ENCODING}
--rnn_type=${RNN_TYPE}
--loss_func=${LOSS_FUNC}
--ema_decay=${EMA_DECAY}
--use_hierarchy_selfatt=${USE_HIERARCHY_SELFATT}
--hierarchy_q_pu_att=${HIERARCHY_Q_PU_ATT}
--max_u_len=${MAX_U_LEN}
--dial_encode=${DIAL_ENCODE}
--dropout_keep_prob_emb=${DROPOUT_KEEP_PROB_EMB}
--flat_q_c_att=${FLAT_Q_C_ATT}
--char_embed_size=${CHAR_EMBED_SIZE}
--word_embed_size=${WORD_EMBED_SIZE}
--topM_for_acc_prob=${TOPM_FOR_ACC_PROB}
--topK_list=${TOPK_LIST}
--decode_q=${DECODE_Q}
--decode_p=${DECODE_P}
--char_emb_file=${CHAR_EMB_FILE}
--word_emb_file=${WORD_EMB_FILE}
--token_key_to_use=${TOKEN_KEY_TO_USE}
--pos_weight=${POS_WEIGHT}
--flat_c_q_att=${FLAT_C_Q_ATT}
--max_p_len=${MAX_P_LEN}
--max_q_len=${MAX_Q_LEN}
--context_window=${CONTEXT_WINDOW}
--question_window=${QUESTION_WINDOW}
--acc_sum_prob=${ACC_SUM_PROB}
--max_num_char_to_keep_forward=${MAX_NUM_CHAR_TO_KEEP_FORWARD}
--max_num_char_to_keep_backward=${MAX_NUM_CHAR_TO_KEEP_BACKWARD}
--num_filters=${NUM_FILTERS}
--evaluate
--algo=${ALGO}
--train_embedding=${TRAIN_EMBEDDING}
--use_char_embedding=${USE_CHAR_EMBED}
--use_selfatt=${USE_SELFATT}
--hidden_size=${HIDDEN_SIZE}
--learning_rate=${LEARNING_RATE}
--weight_decay=${WEIGHT_DECAY}
--batch_size=${BATCH_SIZE}
--vocab_dir=${VOCAB_DIR}
--model_dir=$TRAINING_DIR/models/
--result_dir=$TRAINING_DIR/results_on_test/
--summary_dir=$TRAINING_DIR/summary/
'''

def get_args(config_file):
    CMD = 'source %s; echo "%s"' % (config_file, pargs)
    p = subprocess.Popen(CMD, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    #pargs_str = p.stdout.readlines()[0].strip()
    pargs_str = ''.join(p.stdout.readlines()).replace('\n',' ').strip()
    parser = get_parser()
    return parser.parse_args(pargs_str.split(' '))

def get_parser():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Dialogue System on Therapy Observer')

    def comma_sep_float_list(comma_str):
        return [float(i_str) for i_str in comma_str.split(',') if i_str !='']

    def comma_sep_int_list(comma_str):
        return [int(i_str) for i_str in comma_str.split(',') if i_str !='']

    def comma_sep_str_list(comma_str):
        return [i_str for i_str in comma_str.split(',') if i_str !='']

    command_settings = parser.add_argument_group('command settings')
    command_settings.add_argument(
        '--prepare',
        action='store_true',
        help='create the directories, prepare the vocabulary and embeddings')
    command_settings.add_argument(
        '--train',
        action='store_true',
        help='train the model')
    command_settings.add_argument(
        '--evaluate', action='store_true',
        help='evaluate the model on dev set')
    command_settings.add_argument(
        '--pipe', action='store_true',
        help='evaluate the model ')
    command_settings.add_argument(
        '--predict',
        action='store_true',
        help='predict the answers for test set with trained model')
    command_settings.add_argument(
        '--ensemble',
        action='store_true',
        help='predict the answers by ensembling multiple predicted outputs')
    parser.add_argument('--gpu', type=str, default='',
                        help='specify gpu device')

    parser.add_argument('--cluster_strategy', choices=['MISC11_WOE','MISC15_WOE','MISC28_WOE','MISC11_EL','MISC15_EL','MISC28_EL','MISC11_ML','MISC15_ML','MISC28_ML'], required=True, help='how to cluster the MISC codes, EL means using exception label, ML means when exception, using majority label')

    # train setting, include loss, optimizer,
    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0.00005,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.8,
                                help='dropout keep rate')
    train_settings.add_argument('--dropout_keep_prob_emb', type=float, default=0.8,
                                help='dropout keep rate for embedding')
    train_settings.add_argument('--dropout_keep_prob_mlp', type=float, default=0.8,
                                help='dropout keep rate for output mlp layers')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--pos_ratio', type=float, default=0.20,
                                help='ratio for positive example in a mini batch')
    train_settings.add_argument('--pos_weight', type=float, default=0.5,
                                help='weight for positive examples for binary cross entropy loss')
    train_settings.add_argument('--focal_loss_gama', type=comma_sep_float_list, default='0,0,0,0,0,0',
                                help='power for penalty easy examples for focal cross entropy loss, every element for one label, make sure it is the same with num_classes')
    train_settings.add_argument('--loss_weights', type=comma_sep_float_list, default='1.0,1.0,1.0',
                                help='the alpha in weighted ce, and weighted focal loss')
    train_settings.add_argument('--epochs', type=int, default=20,
                                help='train epochs')
    train_settings.add_argument('--max_gradient_norm', type=float, default=5.0,
                                help='perform gradient clipping')
    train_settings.add_argument('--ema_decay', type=float,
                                default=0.0, help='exponential moving average decay')
    train_settings.add_argument('--loss_func', type=str,
                                default='XENTROPY',
                                help='choose the loss_func to use X_ENTROPY, PAIRWISE_HINGE, PAIRWISE_HINGE@K, PW_SCORE_HINGE, STRUCT_X_ENTROPY')
    # evaluation settings.
    eval_settings = parser.add_argument_group('evaluation settings')

    eval_settings.add_argument('--topK_list', type=comma_sep_int_list, default="1,2,3,5,10",
                               help='a comma sepatated K value list for recall@K metrics')
    eval_settings.add_argument('--ensemble_predicted_files', type=comma_sep_str_list, default=[],
                               help='a comma sepatated ensemble predicted files for ensembling')
    eval_settings.add_argument('--ensemble_method', choices = ['RANK_SUM', 'VOTE_TOP'], default='VOTE_ITER',
                               help='the ensemble method used to ensemble predicted results')
    eval_settings.add_argument('--topM_for_acc_prob', type=int, default=10,
                               help='topM for accumulated probability for evaluation')
    eval_settings.add_argument('--acc_sum_prob', type=float, default=0.9,
                               help='accumulated probability for evaluation in topM')
    eval_settings.add_argument('--steps_per_checkpoint', type=int, default=1000,
                               help='Number of minibatches step to evaluate the dev set')

    # model settings
    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM', 'GMLSTM', 'RNET', 'GA', 'CGA', 'CGMLSTM','MEMNET'],
                                default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--snt_att_algo', type = str,
                                default='',
                                help='choose the snt-lvl attention algorithm to use')
    model_settings.add_argument('--rnn_type', choices=['bi-lstm', 'bi-gru', 'bi-rnn', 'lstm', 'gru', 'rnn'],
                                default='bi-lstm',
                                help='choose the rnn_type to use')
    model_settings.add_argument('--snt_rnn_type', choices=['bi-lstm', 'bi-gru', 'bi-rnn', 'lstm', 'gru', 'rnn'],
                                default='bi-lstm',
                                help='choose the snt_rnn_type to use')
    model_settings.add_argument('--num_att_head', type=int, default=3,
                                help='num of heads for snt seq mulitple head attention')
    model_settings.add_argument('--snt_seq_hops', type=int, default=2,
                                help='num of hops for snt seq mulitple head attention')
    model_settings.add_argument('--use_shared_encoding', type=bool, default=False,
                                help='use shared encoding for different encoding layer?')
    model_settings.add_argument('--dial_encode', choices=['CONCAT', 'HIERARCHY'],
                                default='CONCAT',
                                help='choose the method to encode dialogue')
    model_settings.add_argument('--decode_func', choices=['WX_B', 'FC','BILINEAR'],
                                default='FC',
                                help='choose the method to encode dialogue')
    model_settings.add_argument('--decode_goal', type = str,
                                default='ALL_LABEL',
                                help='choose the goal to deocde dialogue')
    model_settings.add_argument('--decode_inputs', type=comma_sep_str_list,
                                default=['p_final'],
                                help='choose the input to decode similarities')
    model_settings.add_argument('--decode_combine', type=str,
                                default='additive',
                                help='how to combine each input')
    model_settings.add_argument('--decode_r_with_flatten_pu_labels', type=bool,
                                default=False,
                                help='whether to decode r with flatten_pu_labels')
    model_settings.add_argument('--token_key_to_use', choices=['tokenized_utterance', 'norm_tokenized_utterance', 'expanded_tokenized_utterance'],
                                default='tokenized_utterance',
                                help='choose the token key to use')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=60,
                                help='max length of answer')
    model_settings.add_argument('--max_u_len', type=int, default=60,
                                help='max length of a single utterance')
    model_settings.add_argument('--context_window', type=int, default=4,
                                help='window size of the context, x[-n:]')
    model_settings.add_argument('--question_window', type=int, default=1,
                                help='window size of the question, x[-n:]')
    model_settings.add_argument('--use_selfatt', type=bool, default=False,
                                help='Add self-attention layer on top of the matching layer?')
    model_settings.add_argument('--use_batch_norm', type=bool, default=False,
                                help='whether to use batch_norm layer?')
    model_settings.add_argument('--use_hierarchy_selfatt', type=bool, default=False,
                                help='Add self-attention layer on top of the hierarchy_matching layer? only can be used when dial_encode=HIERARCHY')
    # different matching layer
    model_settings.add_argument('--flat_c_q_att', type=bool, default=True,
                                help='Add flat context-aware question matching layer?')
    model_settings.add_argument('--flat_q_c_att', type=bool, default=True,
                                help='Add flat question-aware context matching layer?')
    model_settings.add_argument('--flat_c_r_att', type=bool, default=True,
                                help='Add flat context-aware response matching layer?')
    model_settings.add_argument('--flat_r_c_att', type=bool, default=True,
                                help='Add flat reponse-aware context matching layer?')
    model_settings.add_argument('--hierarchy_q_pu_att', type=bool, default=True,
                                help='Add hierarchy matching layer, match q with each utterance in u')
    model_settings.add_argument('--hierarchy_r_pu_att', type=bool, default=True,
                                help='Add hierarchy matching layer, match r with each utterance in u')
    model_settings.add_argument('--decode_p', type=bool, default=True,
                                help='Add p encodes to final decode?')
    model_settings.add_argument('--decode_q', type=bool, default=True,
                                help='Add q encodes to final decode?')
    model_settings.add_argument('--decode_r', type=bool, default=True,
                                help='Add r encodes to final decode?')

    model_settings.add_argument('--restore', type=bool, default=False,
                                help='Restore a model and continue training')
    model_settings.add_argument('--use_char_embedding', type=bool, default=False,
                                help='Use character embedding or not')
    model_settings.add_argument('--use_speaker_embedding', type=bool, default=False,
                                help='Use speaker embedding or not')
    model_settings.add_argument('--use_label_embedding', type=bool, default=False,
                                help='Use label embedding or not')
    model_settings.add_argument('--use_concat_p', type=bool, default=False,
                                help='Use concatenated passage or not.')
    model_settings.add_argument('--use_response_u', type=bool, default=False,
                                help='Use response utterance or not.')
    model_settings.add_argument('--use_response_s', type=bool, default=False,
                                help='Use response speaker or not.')
    model_settings.add_argument('--use_r_in_seq', type=bool, default=False,
                                help='Use response utterance and speaker info in the context seq for snt attention and aggeragation')
    model_settings.add_argument('--use_r_in_wm_seq', type=bool, default=False,
                                help='Use response utterance and speaker info in the context seq for word match, including self attention')
    model_settings.add_argument('--max_num_char_to_keep_forward', type=int, default=4,
                                help='The maximum number of forward characters used for a word during character RNN training')
    model_settings.add_argument('--max_num_char_to_keep_backward', type=int, default=4,
                                help='The maximum number of backward characters used for a word during character RNN training')
    model_settings.add_argument('--char_embed_size', type=int, default=100,
                                help='size of character embeddings')
    model_settings.add_argument('--speaker_embed_dim', type=int, default=8,
                                help='size of speaker embeddings')
    model_settings.add_argument('--label_embed_dim', type=int, default=32,
                                help='size of label embeddings')
    model_settings.add_argument('--word_embed_size', type=int, default=300,
                                help='size of word embeddings')
    model_settings.add_argument('--filter_sizes', type=comma_sep_int_list, default="3,4,5",
                                help='Comma-separated filter sizes')
    model_settings.add_argument('--num_filters', type=int, default=128,
                                help='Number of filters per filter size')
    model_settings.add_argument('--num_classes', default=16,
                                help='Number of classes in classification')
    model_settings.add_argument('--gated_memnet', type=bool, default=False,
                                help='Use the gated version of memnet')
    model_settings.add_argument('--passage_hops', type=int, default=2,
                                help='number of hops for passage')
    model_settings.add_argument('--response_hops', type=int, default=2,
                                help='number of hops for response')
    model_settings.add_argument('--memnet_share_weights', type=bool, default=True,
                                help='If true, response memnet uses same parameters as the passage memnet')

    # path_settings includes training, dev, test, and paraphrase files
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', type=comma_sep_str_list,
                               default=['../data/demo/trainset/zhidao.train_4500.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', type=comma_sep_str_list,
                               default=['../data/demo/devset/zhidao.dev_500.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', type=comma_sep_str_list,
                               default=['../data/demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')

    # allow to use embedding
    path_settings.add_argument('--word_emb_file', default=None,
                               help='pretrained word embedding')

    path_settings.add_argument('--char_emb_file', default=None,
                               help='pretrained char embedding')
    model_settings.add_argument('--train_embedding', type=bool, default=True,
                                help='Train word embedding or not')
    model_settings.add_argument('--train_speaker_embedding', type=bool, default=True,
                                help='Train speaker embedding or not')
    model_settings.add_argument('--train_label_embedding', type=bool, default=True,
                                help='Train label embedding or not')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='models/',
                               help='the dir to store models')
    path_settings.add_argument('--p_model_config', default='',
                               help='configfile for P model')
    path_settings.add_argument('--t_model_config', default='',
                               help='config file for T model')
    path_settings.add_argument('--model_prefix_to_restore', default='',
                               help='the prefix to restore models')
    path_settings.add_argument('--result_dir', default='results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    elmo_settings = parser.add_argument_group('elmo settings')
    elmo_settings.add_argument('--use_character_elmo', type=bool, default=True,
                               help='indicator whether character elmo embeddings are utilized')
    elmo_settings.add_argument('--elmo_positions', type=comma_sep_str_list, default=[],
                               help='list of positions to add elmo embeddings')
    elmo_settings.add_argument('--elmo_weight_file', default='elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',
                               help='elmo pretrained LM weight file')
    elmo_settings.add_argument('--elmo_option_file', default='elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
                               help='elmo weight corresponding option file')
    elmo_settings.add_argument('--elmo_vocab_file', default='elmo_vocab_file',
                               help='elmo vocabulary file')
    elmo_settings.add_argument('--elmo_snt_dict_file', default='elmo_snt_dict_file.with',
                               help='elmo snt dict file')
    elmo_settings.add_argument('--elmo_u_cache_file', default='',
                               help='elmo utterance cache file path')
    elmo_settings.add_argument('--elmo_p_cache_file', default='',
                               help='elmo passage cache file')
    elmo_settings.add_argument('--elmo_q_cache_file', default='',
                               help='elmo question cache file')
    elmo_settings.add_argument('--elmo_emb_output', type=int, default=128,
                               help='elmo emb output size to be projected')
    return parser
