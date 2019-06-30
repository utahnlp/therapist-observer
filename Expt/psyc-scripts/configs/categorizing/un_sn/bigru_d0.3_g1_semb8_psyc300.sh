# the data dir contains those corpus
#this folder will be created, and all the models and results will be list there.
CONFIG_DIR=/categorizing/un_sn/bigru_d0.3_g1_semb8_psyc300
WORK_DIR=$ROOT_DIR/Expt/workdir/$CONFIG_DIR/
TRAINING_DIR=$WORK_DIR/training/

# cluster strategy for psyc dataset
CLUSTER_STRATEGY=MISC11_ML
INPUT_DIR=psyc_${CLUSTER_STRATEGY}_17_padding

# use pretrained word and char embedding using prepare.sh
VOCAB_DIR=$DATA_DIR/vocab_psyc300_pred_un_sn/

mkdir -p $VOCAB_DIR

# Use ../../utils/preprocess.py to compute score for each paragraph

#TRAIN_FILENAME1=${INPUT_DIR}/train_sc20.json
TRAIN_FILENAME1=${INPUT_DIR}/train.json
#DEV_FILENAME1=${INPUT_DIR}/dev_sc10.json
DEV_FILENAME1=${INPUT_DIR}/dev.json
TEST_FILENAME1=${INPUT_DIR}/test.json
#TEST_FILENAME1=${INPUT_DIR}/dev_sc20.json
RO_TRAIN_FILE1=$RO_DATA_DIR/$TRAIN_FILENAME1
RO_DEV_FILE1=$RO_DATA_DIR/$DEV_FILENAME1
RO_TEST_FILE1=$RO_DATA_DIR/$TEST_FILENAME1
TRAIN_FILE1=$DATA_DIR/prep_data/rmstop_0_rpt_no/$TRAIN_FILENAME1
DEV_FILE1=$DATA_DIR/prep_data/rmstop_0_rpt_no/$DEV_FILENAME1
TEST_FILE1=$DATA_DIR/prep_data/rmstop_0_rpt_no/$TEST_FILENAME1


ALGO="GMLSTM"
LEARNING_RATE=0.0001
#WEIGHT_DECAY=0.0001
WEIGHT_DECAY=0.0
BATCH_SIZE=128
CONTEXT_WINDOW=1
QUESTION_WINDOW=0
HIDDEN_SIZE=64
EPOCH=100
STEPS_PER_CHECKPOINT=100
# DROP_KEPP_PROB in (0, 1], 1 is no dropout
DROP_KEEP_PROB=0.7
USE_SELFATT=
USE_CHAR_EMBED=
MAX_NUM_CHAR_TO_KEEP_FORWARD=4
MAX_NUM_CHAR_TO_KEEP_BACKWARD=4
#USE_CHAR_EMBED=x
# Whether to training the original embedding.
TRAIN_EMBEDDING=
# max_grad_norm / max(global_norm, max_grad_norm), set to inf to disable.
MAX_GRAD_NORM=1.0
# leave it empty to use random initial WORD_EMB
# WORD_EMB_FILE=$RO_DATA_DIR/glove.840B.300d.txt
WORD_EMB_FILE=$RO_DATA_DIR/psyc_glove_300d.txt
# WORD_EMB_FILE=
#WORD_EMB_FILE=$DATA_DIR/vectors_words.txt
CHAR_EMB_FILE=
#CHAR_EMB_FILE=$RO_DATA_DIR/glove.840B.300d-char.txt
#CHAR_EMB_FILE=$DATA_DIR/vectors_chars.txt
EMA=0.9999
MAX_P_LEN=1000
MAX_Q_LEN=60
NUM_FILTERS=25
ACC_SUM_PROB=0.9
#flat Context-aware question attention
FLAT_C_Q_ATT=
# pos_weight for balanced cross entropy
POS_WEIGHT=0.9
# set gama = 0, decay to standard cross entropy
# key for tokenization to use
TOKEN_KEY_TO_USE=tokenized_utterance
# whether adding p encoding to decode
DECODE_P=
# whether adding q encoding to decode
DECODE_Q=
# TOPK, a list of integers for K values in Recall@K
TOPK_LIST=1,2,3,5,10
# TOPM_FOR_ACC_PROB, with ACC_SUM_PROB in topM
TOPM_FOR_ACC_PROB=5
# WORD_EMBED_SIZE, default 300, exclusive with WORD_EMB_FILE
WORD_EMBED_SIZE=300
# CHAR_EMBED_SIZE, default 100, exclusive with CHAR_EMB_FILE
CHAR_EMBED_SIZE=300
# flat Question-aware context attention
FLAT_Q_C_ATT=
# Dropout keep prob for embedding, 1.0=no_dropout
DROPOUT_KEEP_PROB_EMB=0.7
# Method to encode the dialogue
DIAL_ENCODE=CONCAT
# max_length for a single utterance
MAX_U_LEN=60
# whether to hierarchy_q_pu_att
HIERARCHY_Q_PU_ATT=
# self-att for hierarchy, only can be useful when dial_encode=HIERARCHY
USE_HIERARCHY_SELFATT=
# ema_decay is decay ratio for EMA, 0.0 to disable, 0.9999+ to enable
EMA_DECAY=0.0
# loss_func, default=X_ENTROPY
LOSS_FUNC=X_ENTROPY
# rnn_type, bi-lstm, bi-gru, bi-rnn, lstm, gru, rnn
RNN_TYPE=bi-gru
# whether to use shared encode layer for utterance
USE_SHARED_ENCODING=
# all training files to use
TRAIN_FILES=$TRAIN_FILE1
#TRAIN_FILES=`find ${TRAIN_FILE1}_splits -name "split*" | tr '\n' ','`
# all dev files to use
DEV_FILES=$DEV_FILE1
# all test files to use
TEST_FILES=$TEST_FILE1
# pw_alpha * pairwise_hinge_loss + x_entropy
# indicator whether character elmo embeddings are utilized
USE_CHARACTER_ELMO=x
# list of positions names to add elmo embeddings
ELMO_POSITIONS=
# elmo pretrained LM weight file
ELMO_WEIGHT_FILE=$DATA_DIR/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
# elmo corresponding to options file
ELMO_OPTION_FILE=$DATA_DIR/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
# elmo vocabulary file
ELMO_VOCAB_FILE=$DATA_DIR/elmo_vocab_file
# elmo max num character 
# elmo embedding output size to be projected into
ELMO_EMB_OUTPUT=128
# whether use character elmo emebdding
USE_CHARACTER_ELMO=x
# positions to inject elmo, keep empty to disable
ELMO_POSITIONS=
# elmo option json file
ELMO_OPTION_FILE=$DATA_DIR/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
# elmo vocabulary file to write and read
ELMO_VOCAB_FILE=$DATA_DIR/elmo_vocab_file
# elmo max num of characters
# input used to decode
DECODE_INPUTS=r_final

#MEMNET PARAMS
GATED_MEMNET=x
PASSAGE_HOPS=
MEMNET_SHARE_WEIGHTS=x


# whether to use concat p
USE_CONCAT_P=
# decode_func to use for multiclass decoding
DECODE_FUC=FC
# flat Context-ware response attention
FLAT_C_R_ATT=
# flat response-ware context attention
FLAT_R_C_ATT=
# whether to hierarchy_r_pu_att
HIERARCHY_R_PU_ATT=
# whether adding r encoding to cnn decode
DECODE_R=
# r pass memnet hops
RESPONSE_HOPS=2
# use response utterance or not
USE_RESPONSE_U=x
MODEL_PREFIX_TO_RESTORE=GMLSTM
# decode goal
DECODE_GOAL=ALL_LABEL
# Whether to use speaker embedding
USE_SPEAKER_EMBEDDING=x
# Whether to use label embedding
USE_LABEL_EMBEDDING=
# dim of label embedding
LABEL_EMBED_DIM=32
# dim of speaker embedding
SPEAKER_EMBED_DIM=8
# filter sizes for cnn
FILTER_SIZES=3,4,5
# whether to decode r with flatten pu_labels embdding
DECODE_R_WITH_FLATTEN_PU_LABELS=
# whether to use response speaker info
USE_RESPONSE_S=x
# whether to train speaker emb
TRAIN_SPEAKER_EMBEDDING=x
# whether to train label emb
TRAIN_LABEL_EMBEDDING=x
# dropout keep rate for MLP
DROPOUT_KEEP_PROB_MLP=0.8
# num_attention_heads for snt seq attention
NUM_ATT_HEAD=4
# snt-levl attention algorithm, leave empty for disabling
SNT_ATT_ALGO=
# snt-levl attention hops
SNT_SEQ_HOPS=2
# snt rnn type, for snt-lvl rnn
SNT_RNN_TYPE=bi-gru
# loss_weights for each label, sep with comma, can be float
LOSS_WEIGHTS=1.0,1.0,0.25,0.75,0.75,0.25,0.75,1.0,1.0,1.0,1.0
# focal loss gama for each label, sep with comma, int
FOCAL_LOSS_GAMA=0,0,0,0,0,0,0,0,0,0,0
# use response in context seq, without its label
USE_R_IN_SEQ=
# how to combine the final input states
DECODE_COMBINE=additive
#config file for P model
P_MODEL_CONFIG=
#config file for T model
T_MODEL_CONFIG=
# whether use r in word matching
USE_R_IN_WM_SEQ=
# whether use batch normalization
USE_BATCH_NORM=
