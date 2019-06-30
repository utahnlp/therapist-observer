# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 14:59:10 jcao>
# --------------------------------------------------------------------
# File Name          : dial_model.py
# Original Author    : jiessie.cao@gmail.com
# Description        : This module implements the reading comprehension models based on:
# code skeleton borrowed from
# https://github.com/baidu/DuReader/blob/master/tensorflow/rc_model.py
# 1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
# 2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
# Note that we use Pointer Network for the decoding stage of both models.
# --------------------------------------------------------------------

import os
import time
import logging
import re
import ujson as json
import numpy as np
import copy
import tensorflow as tf
from tqdm import tqdm
import math_utils
import tf_utils
from psyc_utils import PsycUtils
from elmo_utils import ELMo_Utils
import loss_zoo
from layers.basic_rnn import rnn
#from layers.memory_network import memory_network_rnn_multi_mem
#from layers.gated_memory_network import gated_memory_network_rnn_multi_mem
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.match_layer import GatedAttentionLayer
from layers.match_layer import CollaborativeGatedAttentionLayer
from layers.match_layer import GatedMatchLSTMLayer
from layers.match_layer import CollaborativeGatedMatchLSTMLayer
from layers.match_layer import multihead_attention, dot_product_attention, scaled_dot_product_attention
# from layers.pointer_net import PointerNetDecoder


class DialogueModel(object):
    """
    Implements the main dialogue model according to the preivous rc_model.
    """
    def __init__(self, vocab, vocab_char, args, elmo_utils=None):
        # logging
        self.logger = logging.getLogger("dial")

        # multi_head attention
        self.num_att_head = args.num_att_head

        # snt_attention
        self.snt_att_algo = args.snt_att_algo
        self.snt_seq_hops = args.snt_seq_hops
        self.use_snt_att = True if self.snt_att_algo else False

        # basic config
        self.algo = args.algo
        # default rnn_type = self.rnn_type
        self.rnn_type = args.rnn_type
        self.snt_rnn_type = args.snt_rnn_type
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.decode_inputs = args.decode_inputs
        self.decode_func = args.decode_func
        self.decode_goal = args.decode_goal
        self.decode_combine = args.decode_combine

        # context_window for label contacting dim
        self.context_window = args.context_window

        self.use_r_in_seq = args.use_r_in_seq
        self.use_r_in_wm_seq = args.use_r_in_wm_seq

        # speaker embedding:
        self.speaker_embed_dim = args.speaker_embed_dim

        # label embedding:
        self.label_embed_dim = args.label_embed_dim
        self.psyc_utils = PsycUtils(args.cluster_strategy)

        if 'SPEAKER' in self.decode_goal:
            self.classes = PsycUtils.SPEAKERS
            self.brief_classes = PsycUtils.SPEAKERS
        elif 'ALL_LABEL' in self.decode_goal:
            self.classes = self.psyc_utils.All_labels
            self.brief_classes = self.psyc_utils.brief_All_labels
        elif 'P_LABEL' in self.decode_goal:
            self.classes = self.psyc_utils.P_labels
            self.brief_classes = self.psyc_utils.brief_P_labels
        elif 'T_LABEL' in self.decode_goal:
            self.classes = self.psyc_utils.T_labels
            self.brief_classes = self.psyc_utils.brief_T_labels
        else:
            raise NotImplementedError("{} is not supported".format(self.decode_goal))

        # for seq taging ,also add padding tag
        if 'SEQTAG' in self.decode_goal:
            self.classes = self.classes + [PsycUtils.PADDING_LABEL]
            self.brief_classes = self.brief_classes + [PsycUtils.PADDING_LABEL]
 
        self.num_classes = len(self.classes)
        self.pred_classes_mapping = {}

        for i, x in enumerate(self.classes):
            self.pred_classes_mapping[x] = i

        self.use_concat_p = args.use_concat_p
        self.question_window = args.question_window
        self.use_question = args.question_window > 0
        self.use_response_u = args.use_response_u
        self.use_response_s = args.use_response_s

        # use dropout for other layers
        self.use_dropout = args.dropout_keep_prob < 1
        # use dropout for embedding
        self.use_dropout_emb = args.dropout_keep_prob_emb < 1
        # use dropout for MLP
        self.use_dropout_mlp = args.dropout_keep_prob_mlp < 1

        # whether use shared encoding weights for bi-lstm
        self.use_shared_encoding = args.use_shared_encoding


        # loss function ,choice = ['X_ENTROPY', 'WEIGHTED_CE', 'WEIGHTED_FOCAL']
        self.loss_func = args.loss_func
        self.loss_weights = args.loss_weights
        if self.loss_func in ['WEIGHTED_CE','WEIGHTED_FOCAL']:
            assert len(self.loss_weights) == self.num_classes

        # for balanced cross entropy and focal loss
        self.pos_weight = args.pos_weight
        self.focal_loss_gama = args.focal_loss_gama

        if self.loss_func == 'WEIGHTED_FOCAL':
            assert len(self.focal_loss_gama) == self.num_classes

        # matching layers
        # count of flat matching layer is also the output group size of the cnn layers.
        self.flat_c_q_att = args.flat_c_q_att
        self.flat_c_r_att = args.flat_c_r_att
        self.flat_q_c_att = args.flat_q_c_att
        self.flat_r_c_att = args.flat_r_c_att

        # matching used for hierarchy lstm
        self.hierarchy_q_pu_att = args.hierarchy_q_pu_att
        self.hierarchy_r_pu_att = args.hierarchy_r_pu_att
        # dial_encode can be CONCAT and HIERARCHY
        # CONCAT means concat the utterances in the conversation message
        # HIERARCHY means the hierarchy lstm to mode the conversation message
        self.dial_encode = args.dial_encode

        self.dropout_keep_prob_value = args.dropout_keep_prob
        self.dropout_keep_prob_emb_value = args.dropout_keep_prob_emb
        self.dropout_keep_prob_mlp_value = args.dropout_keep_prob_mlp

        #memory network params
        self.gated_memnet = args.gated_memnet
        self.passage_hops = args.passage_hops
        self.response_hops = args.response_hops
        self.memnet_share_weights = args.memnet_share_weights

        # see wether the output lstm encoding of p, q, a directly used as in the cnn_pooling input
        self.decode_p = args.decode_p
        self.decode_q = args.decode_q
        self.decode_r = args.decode_r

        # True is 1
        self.att_group_size = sum([self.flat_c_q_att, self.flat_q_c_att, self.flat_c_r_att, self.flat_r_c_att])
        self.decode_group_size = sum([self.decode_p, self.decode_q, self.decode_r]) + self.att_group_size

        # self-attention layers, switch for flat word-level self attention
        self.use_selfatt = args.use_selfatt
        # hierarchy self-attention layers, switch for hierarchy sentence-level self attention
        self.use_hierarchy_selfatt = args.use_hierarchy_selfatt

        # gradient norm and clipping gradient
        self.max_gradient_norm = args.max_gradient_norm
                # every n steps, do evaluation on dev set, and do checkpoint.
        self.steps_per_checkpoint = args.steps_per_checkpoint
        # EMA Decay ratio.
        self.use_ema = args.ema_decay > 0
        self.use_batch_norm = args.use_batch_norm

        # length limit
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_u_len = args.max_u_len
        self.max_passage_window = self.context_window

        # the vocab
        self.vocab = vocab
        self.vocab_char = vocab_char

        self.use_char_embedding = args.use_char_embedding
        self.use_speaker_embedding = args.use_speaker_embedding
        self.use_label_embedding = args.use_label_embedding
        self.decode_r_with_flatten_pu_labels = args.decode_r_with_flatten_pu_labels
        self.train_embedding = args.train_embedding
        self.train_speaker_embedding = args.train_speaker_embedding
        self.train_label_embedding = args.train_label_embedding

        # the CNN
        self.filter_sizes = args.filter_sizes
        self.num_filters = args.num_filters
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.state_size = args.hidden_size * 2
        self.encode_output_size = 2 * self.state_size if self.use_char_embedding else self.state_size
        # for pool(cnn) sentence representation, the output feature size is cnn group size*num_filter_total
        # @TODO: what if pool(Hash(Ngram(x)))
        self.feature_size = self.decode_group_size * self.num_filters_total

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
	sess_config.allow_soft_placement=True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=sess_config)

        self.epoch_number = args.epochs

        # for elmo
        self.elmo_utils = elmo_utils
        self.elmo_positions = args.elmo_positions
        self.use_elmo = True if args.elmo_positions else False
        self.elmo_emb_output = args.elmo_emb_output

        with self.graph.as_default():
            # current global steps (one whole batch is one ste, it is the number of batches proprocessed)
            self.global_step = tf.get_variable(
                'global_step',
                shape=[],
                dtype=tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False)

            self._build_graph()

            if self.use_ema:
                self.logger.warning('Using Exp Moving Average to train the model with decay {}.'.format(args.ema_decay))
                self.args = args
                self.ema = tf.train.ExponentialMovingAverage(decay=args.ema_decay, num_updates=self.global_step)
                self.ema_op = self.ema.apply(self.all_params)
                with tf.control_dependencies([self.train_op]):
                    self.train_op = tf.group(self.ema_op)
                self.ema_test_graph = tf.Graph()
                self.ema_test_model = None

            # save info
            self.saver = tf.train.Saver(max_to_keep=100)
            self.sess.run(tf.global_variables_initializer())
        # tf.get_default_graph().finalize()

    # All the encode layers
    def _encode(self):
        if self.use_shared_encoding:
            if self.use_concat_p:
                self._encode_p(scope_name = 'shared')
            if self.use_response_u:
                self._encode_r(scope_name = 'shared')
            if self.use_question:
                self._encode_q(scope_name = 'shared')
        else:
            if self.use_concat_p:
                self._encode_p(scope_name = 'passage')
            if self.use_response_u:
                self._encode_r(scope_name = 'response')
            if self.use_question:
                self._encode_q(scope_name = 'question')

    def _low_encode(self):
        if self.use_shared_encoding:
            self._encode_pu(scope_name = 'low_shared')
        else:
            self._encode_pu(scope_name = 'low_passage')

    def _hierarchy_encode(self):
        # hierachy encoding
        self._low_encode()
        # hierarchy word att
        if self.hierarchy_q_pu_att or self.hierarchy_r_pu_att:
            self._hierarchy_match()
            if self.use_hierarchy_selfatt:
                self._hierarchy_selfatt()
            self._hierarchy_fuse()
        # hierarchy high sentence-level encoding
        self._high_encode()

    def _word_match_for_concated(self):
        """
        word level attention between concatenated rep.
        """
        if self.att_group_size:
            self._match()
            if self.use_selfatt:
                self._selfatt()
            self._fuse()

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        if 'input' in self.elmo_positions:
            if self.elmo_utils:
                self._elmo_embed_input_with_cache()

        if self.algo == 'MEMNET':
            # self._run_memory_network(gated=self.gated_memnet)
            raise NotImplementedError("self.algo {} is not implemented".format(self.algo))
        else:
            # encode layers
            if self.dial_encode == 'CONCAT':
                self._encode()
                self._word_match_for_concated()
            elif self.dial_encode == 'HIERARCHY':
                # for now, we still use the concated encoding at the same time
                self._encode()
                # hierarchy encode
                self._hierarchy_encode()
                self._word_match_for_concated()
            else:
                raise NotImplementedError("dial_encode {} is not implemented".format(self.dial_encode))

        if 'SEQTAG' in self.decode_goal:
            self._decode_seq_tags()
        else:
            self._decode_multiclass()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.train:
            with tf.control_dependencies(update_ops):
                self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders_for_p(self):
        if self.use_concat_p:
            # [batch_size, max_length of the content]
            self.p = tf.placeholder(tf.int32, [None, None], name='p')
            # token length for p, [batch_size]
            self.p_length = tf.placeholder(tf.int32, [None], name='p_length')
            # [batch_size, max_length]
            self.p_mask = tf.sequence_mask(self.p_length, tf.shape(self.p)[-1], dtype=tf.float32)
            if self.use_char_embedding:
                self.pc = tf.placeholder(tf.int32, [None, None], name='pc')  # Haohui
                # char length for [batch_size * max_length]
                self.p_t_length = tf.placeholder(tf.int32, [None], name='p_t_length')  # Jie

    # it is for concatnated q, given question window, we also can use the pu for question repreentation.
    def _setup_placeholders_for_q(self):
        # [batch_size, max_length of the content]
        self.q = tf.placeholder(tf.int32, [None, None], name='q')
        # token length for q, [batch_size]
        self.q_length = tf.placeholder(tf.int32, [None], name='q_length')
        # [batch_size * self.num_classes, max_length]
        self.q_mask = tf.sequence_mask(self.q_length, tf.shape(self.q)[-1], dtype=tf.float32)
        # [batch_size * self.max_passage_window]
        self.dup_q_for_pu_length = tf_utils._tf_dup_1(self.q_length, self.max_passage_window)
        # [batch_size * self.max_passage_window, max_length]
        self.dup_q_for_pu_mask = tf.sequence_mask(self.dup_q_for_pu_length, tf.shape(self.q)[-1], dtype=tf.float32)
        if self.use_char_embedding:
            # [batch_size * max_length, max_char_length]
            # when contacting to word_embeding, it should be reshaped
            self.qc = tf.placeholder(tf.int32, [None, None], name='qc')
            # char length for [batch_size * max_length]
            self.q_t_length = tf.placeholder(tf.int32, [None], name='q_t_length')

    def _setup_placeholders_for_r(self):
        # [batch_size, max_length of the content]
        self.r = tf.placeholder(tf.int32, [None, None], name='r')
        # token length for r, [batch_size]
        self.r_length = tf.placeholder(tf.int32, [None], name='r_length')
        # speaker for every utterance in the passage window
        # [batch_size]
        self.r_speaker = tf.placeholder(tf.int32, [None], name='r_speaker')
        # [batch_size, max_u_length]
        self.r_mask = tf.sequence_mask(self.r_length, tf.shape(self.r)[-1], dtype=tf.float32)
        # dup_r_for_pu_length
        # [batch_size * self.max_passage_window]
        self.dup_r_for_pu_length = tf_utils._tf_dup_1(self.r_length, self.max_passage_window)
        # [batch_size * self.max_passage_window, max_length]
        self.dup_r_for_pu_mask = tf.sequence_mask(self.dup_r_for_pu_length, tf.shape(self.r)[-1], dtype=tf.float32)
        if self.use_char_embedding:
            # [batch_size * max_length, max_char_length]
            # when contacting to word_embeding, it should be reshaped
            self.rc = tf.placeholder(tf.int32, [None, None], name='rc')
            # char length for [batch_size * max_length]
            self.r_t_length = tf.placeholder(tf.int32, [None], name='r_t_length')

    def _setup_placeholders_for_pu(self):
        # [batch_size * max_passage_window_size, max_length_of_utterance]
        self.p_u = tf.placeholder(tf.int32, [None, None], name='p_u')
        # token counts for each utterance in passage window
        # [batch_size * passage_window_size]
        self.p_u_length = tf.placeholder(tf.int32, [None], name='p_u_length')
        # number of utterances for each p
        # [batch_size]
        self.p_wsize = tf.placeholder(tf.int32, [None], name='p_wsize')
        # speaker for every utterance in the passage window
        # [batch_size * max_passage_window_size]
        self.p_u_speaker = tf.placeholder(tf.int32, [None], name='p_u_speaker')
        # label for every utterance in the passage window
        # [batch_size * max_passage_window_size]
        self.p_u_label = tf.placeholder(tf.int32, [None], name='p_u_label')
        # mask for each utterance
        # [batch_size * max_passage_window_size, max_u_length]
        self.p_utterance_mask = tf.sequence_mask(self.p_u_length, tf.shape(self.p_u)[-1], dtype=tf.float32)
        # [batch_size, max_window_passage]
        self.p_window_mask = tf.sequence_mask(self.p_wsize, tf.shape(tf.reshape(self.p_u, [tf.shape(self.p_wsize)[0], -1]))[-1], dtype=tf.float32)
        if self.use_char_embedding:
            self.p_u_c = tf.placeholder(tf.int32, [None, None, None], name='p_u_c')  # Jie
            # char length for [batch_size, passage_window * max_length]
            self.p_u_t_length = tf.placeholder(tf.int32, [None, None], name='p_u_t_length')

    def _setup_placeholders(self):
        """
        Placeholders
        """
        if self.use_concat_p:
            self._setup_placeholders_for_p()

        self._setup_placeholders_for_q()
        self._setup_placeholders_for_pu()
        self._setup_placeholders_for_r()

        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.dropout_keep_prob_emb = tf.placeholder(tf.float32, name='dropout_keep_prob_emb')
        self.dropout_keep_prob_mlp = tf.placeholder(tf.float32, name='dropout_keep_prob_mlp')
        self.is_training_phase = tf.placeholder(tf.bool, name='is_training_phase')
        # we only have batch_size correct labels, one label for each example.
        # when comparing to these correct labels, we need to reshape previous computing into [batch_size, self.num_classes], which is also compatible for binary classifier.
        # [batch_size]
        self.correct_labels = tf.placeholder(tf.int32, [None], name='correct_label')  # Jie
        # [batch_size * max_snt_windows]
        self.correct_seq_labels = tf.placeholder(tf.int32, [None], name='correct_seq_label')  # Jie
        if  self.use_elmo and self.elmo_utils:
            # self.elmo_utils.add_elmo_placeholder_with_cache_sntids()
            self.elmo_utils.add_elmo_placeholder_with_cache_emb()

    def _embed_speaker(self):
        """
        The embedding layer for modeling speaker, which can be the sentence level feature for both sentence classification or hierarchical model.
        """
        if self.train_speaker_embedding:
            const_init_speaker_emb = np.random.rand(self.psyc_utils.num_all_speakers_with_padding, self.speaker_embed_dim)
            # const_init_speaker_emb = np.random.rand(PsycUtils.num_speaker, self.speaker_embed_dim)
        else:
            const_init_speaker_emb = np.eye(self.psyc_utils.num_all_speakers_with_padding, self.speaker_embed_dim)

        with tf.variable_scope('speaker_embedding'):
            self.speaker_embeddings = tf.get_variable(
                'speaker_embeddings',
                shape=(self.psyc_utils.num_all_speakers_with_padding, self.speaker_embed_dim),
                #shape=(PsycUtils.num_speaker, self.speaker_embed_dim),
                initializer=tf.constant_initializer(const_init_speaker_emb),
                trainable=self.train_speaker_embedding
            )

            self.r_speaker_emb = tf.nn.embedding_lookup(self.speaker_embeddings, self.r_speaker)
            # [batch_size * max_passage_windows, dim_speaker_emb]
            self.p_u_speaker_emb = tf.nn.embedding_lookup(self.speaker_embeddings, self.p_u_speaker)

        if self.use_dropout_emb:
            self.r_speaker_emb = tf.nn.dropout(self.r_speaker_emb, self.dropout_keep_prob_emb)
            self.p_u_speaker_emb = tf.nn.dropout(self.p_u_speaker_emb, self.dropout_keep_prob_emb)


    def _embed_label(self):
        """
        The embedding layer for labels, which is the sentence level features for both sentence classification features or hierarchical model
        """
        # when training label, random initial with doubles
        if self.train_label_embedding:
            const_init_label_emb = np.random.rand(self.psyc_utils.num_all_labels_with_padding, self.label_embed_dim)
        else:
            # when fixed embedding, use np.eye
            const_init_label_emb = np.eye(self.psyc_utils.num_all_labels_with_padding, self.label_embed_dim)

        with tf.variable_scope('label_embedding'):
            self.label_embeddings = tf.get_variable(
                'label_embeddings',
                shape=(self.psyc_utils.num_all_labels_with_padding, self.label_embed_dim),
                initializer=tf.constant_initializer(const_init_label_emb),
                trainable=self.train_label_embedding
            )
            # [batch_size * max_passage_windows, dim_label_emb]
            self.p_u_label_emb = tf.nn.embedding_lookup(self.label_embeddings, self.p_u_label)
            # [ label_size, label_dim]
            self.pred_label_emb = tf.nn.embedding_lookup(self.label_embeddings, map(lambda l: self.pred_classes_mapping[l], self.classes))
            pred_label_emb_norm = tf.nn.l2_normalize(self.pred_label_emb, dim = 1)
            self.label_cm = tf.matmul(pred_label_emb_norm, pred_label_emb_norm, transpose_b = True)
        if self.use_dropout_emb:
            self.p_u_label_emb = tf.nn.dropout(self.p_u_label_emb, self.dropout_keep_prob_emb)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        # @TODO: what's the name for a device. What's the usage.
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=self.train_embedding
            )
            if self.use_concat_p:
                # [batch_size, max_len, dim_word_emb]
                self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)

            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)
            self.r_emb = tf.nn.embedding_lookup(self.word_embeddings, self.r)
            # [batch_size * max_passage_windows, max_len, dim_word_emb]
            self.p_u_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p_u)

        if self.use_dropout_emb:
            if self.use_concat_p:
                self.p_emb = tf.nn.dropout(self.p_emb, self.dropout_keep_prob_emb)
            self.p_u_emb = tf.nn.dropout(self.p_u_emb, self.dropout_keep_prob_emb)
            self.q_emb = tf.nn.dropout(self.q_emb, self.dropout_keep_prob_emb)
            self.r_emb = tf.nn.dropout(self.r_emb, self.dropout_keep_prob_emb)

        if self.use_char_embedding:
            with tf.device('/cpu:0'), tf.variable_scope('char_embedding'):
                self.char_embeddings = tf.get_variable(
                    'char_embeddings',
                    shape=(self.vocab_char.size(), self.vocab_char.embed_dim),
                    initializer=tf.constant_initializer(self.vocab_char.embeddings),
                    trainable=True
                )
                if self.use_concat_p:
                    # [batch_size*max_len, max_char_len, dim_char_emb]
                    self.p_emb_char = tf.nn.embedding_lookup(self.char_embeddings, self.pc)
                # [batch_size*max_passage_window*max_len, max_char_len, dim_char_emb]
                self.p_u_emb_char = tf.nn.embedding_lookup(self.char_embeddings, self.p_u_c)
                self.q_emb_char = tf.nn.embedding_lookup(self.char_embeddings, self.qc)
                self.r_emb_char = tf.nn.embedding_lookup(self.char_embeddings, self.rc)

                if self.use_dropout_emb:
                    if self.use_concat_p:
                        self.p_emb_char = tf.nn.dropout(self.p_emb_char, self.dropout_keep_prob_emb)
                    self.p_u_emb_char = tf.nn.dropout(self.p_u_emb_char, self.dropout_keep_prob_emb)
                    self.q_emb_char = tf.nn.dropout(self.q_emb_char, self.dropout_keep_prob_emb)
                    self.r_emb_char = tf.nn.dropout(self.r_emb_char, self.dropout_keep_prob_emb)

        if self.use_speaker_embedding:
            self._embed_speaker()

        if self.use_label_embedding or self.decode_func == 'BILINEAR':
            self._embed_label()


    def _elmo_embed_input_with_cache(self):
        """
        when elmo cache can be used, then use each sample id to retrieve elmo_ops
        """
        self.elmo_utils.elmo_embedding_layer_emb(self.elmo_emb_output)
        # self.elmo_utils.elmo_embedding_layer_sntids(self.elmo_emb_output)

        if self.use_dropout_emb:
            if self.use_concat_p:
                self.elmo_utils.p_elmo_emb = tf.nn.dropout(self.elmo_utils.p_elmo_emb, self.dropout_keep_prob_emb)

            self.elmo_utils.pu_elmo_emb = tf.nn.dropout(self.elmo_utils.pu_elmo_emb, self.dropout_keep_prob_emb)
            self.elmo_utils.q_elmo_emb = tf.nn.dropout(self.elmo_utils.q_elmo_emb, self.dropout_keep_prob_emb)
            self.elmo_utils.r_elmo_emb = tf.nn.dropout(self.elmo_utils.r_elmo_emb, self.dropout_keep_prob_emb)

        #concat word emb amd elmo emb
        if self.use_concat_p:
            self.p_emb = tf.concat([self.p_emb, self.elmo_utils.p_elmo_emb], -1)
        self.p_u_emb = tf.concat([self.p_u_emb, self.elmo_utils.pu_elmo_emb], -1)
        self.q_emb = tf.concat([self.q_emb, self.elmo_utils.q_elmo_emb], -1)
        self.r_emb = tf.concat([self.r_emb, self.elmo_utils.r_elmo_emb], -1)

    def _encode_pu(self, scope_name):
        """
        Employs Bi-LSTMs to encode passage, question, answer separately
        they will use three different vairable_scope.
        """
        with tf.variable_scope('{}_encoding'.format(scope_name), reuse=tf.AUTO_REUSE):
            # p_emb  [batch_size, max_len, dim_word_emb]
            # p_u_emb [batch_size * max_passage_windows, max_len, dim_word_emb]
            # sep_p_u_final_state = [batch_size * max_passage_window, fw_hidden+bw_hidden=2*hidden]
            # sep_p_u_encodes= [batch_size * max_passage_window, max_u_length, fw_hidden+bw_hidden=2*hidden]
            self.sep_p_u_encodes, self.sep_p_u_final_state = rnn(self.rnn_type, self.p_u_emb, self.p_u_length, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
            # put every utterance final state into a sentence-level lstm
            # [batch_size, max_passage_window, fw_hidden + bw_hidden = 2*hidden]
            # self.reshaped_p_utterance_state = tf.reshape(self.sep_p_u_final_state, [tf.shape(self.p_wsize)[0], -1, 2 * self.hidden_size])
            # with tf.vairbale_scope('passage_hierachy_encoding'):
            #    self.sep_p_encodes, self.sep_p_final_state = rnn(self.rnn_type, self.reshaped_p_utterance_state, self.p_wsize, self.hidden_size)

        if self.use_char_embedding:
            with tf.variable_scope('{}_char_encoding'.format(scope_name), reuse = tf.AUTO_REUSE):
                # sep_p_u_encodes_char is alrejdy concated by fw_state and bw_state
                # [batch_size * max_passage_window, 2*hidden]
                _, sep_p_u_encodes_char = rnn(self.rnn_type, self.p_u_emb_char, self.p_u_t_length, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
                # self.p_u is [batch_size * max_passage_window, max_length]
                # [batch_size * max_passage_window * max_length, 2*hidden] can be reshape into [batch_size * max_passage_window, max_length, 2*hidden]
                # after reshaping, then every last state is just aligned to corresponding word embeding.
                self.sep_p_u_encodes_char = tf.reshape(sep_p_u_encodes_char, [-1, tf.shape(self.p_u)[1], 2 * self.hidden_size])

            # [batch_size, max_len, 4 * hidden]
            self.sep_p_u_encodes = tf.concat([self.sep_p_u_encodes, self.sep_p_u_encodes_char], 2)

        if self.use_dropout:
            self.sep_p_u_encodes = tf.nn.dropout(self.sep_p_u_encodes, self.dropout_keep_prob)
            # self.sep_p_u_final_state = tf.nn.dropout(self.sep_p_u_final_state, self.dropout_keep_prob)

#    def _run_memory_network(self, gated = False):
#        """
#        run memory network, merged from @SS
#        """
#        memnet = memory_network_rnn_multi_mem if not gated else gated_memory_network_rnn_multi_mem
#
#        #arep_memnet is (batch_size, rep_dim)
#        batch_size = tf.shape(self.p_wsize)[0]
#        # 2 * hidden_size
#        mem_dim = self.state_size
#        pu = (self.p_u_emb, self.p_u_length, 's_pu', self.max_passage_window)
#
#        with tf.variable_scope('mnet', reuse = tf.AUTO_REUSE):
#            #each entry in the mem seq is a tuple
#            mem_seq_p_u = [pu]
#
#            #[batch_size, passage_window_size, mem_dim]
#            p_u_mem = memnet(
#                    mem_seq = mem_seq_p_u,
#                    input_q = (self.p_u_emb, self.p_u_length),
#                    output_shape = [batch_size, self.max_passage_window, mem_dim],
#                    hops = self.passage_hops,
#                    dropout_keep_prob = self.dropout_keep_prob,
#                    rnn_type = self.rnn_type)
#
#            self.p_u_mrep = p_u_mem.get_final_state()
#            p_u_enc = p_u_mem.get_initial_input_enc()
#            q_enc = p_u_enc[:,-1,:]
#
#            #sum the p_u memory based encodings
#            # p_u_mrep = tf.reduce_sum(p_u_mrep, 1)
#            with tf.variable_scope('hierarchy_passage_encoding'):
#                _, self.p_u_mrep = rnn(self.snt_rnn_type, self.p_u_mrep, self.p_wsize, mem_dim/2, dropout_keep_prob = self.dropout_keep_prob)
#
#        scope_name = 'mnet' if self.memnet_share_weights else 'mnet_r'
#        with tf.device('/device:GPU:1'), tf.variable_scope(scope_name, reuse = True):
#            mem_seq_r = [pu]
#            if self.answers_use_ek:
#                mem_seq_r.append(ek)
#
#            #[batch_size, self.max_candidate_answer, rep_dim]
#            r_mem = memnet(
#                mem_seq = mem_seq_r,
#                input_q = (self.r_emb, self.r_length),
#                output_shape = [batch_size, 1 , mem_dim],
#                hops = self.response_hops,
#                dropout_keep_prob = self.dropout_keep_prob,
#                rnn_type = self.rnn_type)
#
#            self.r_mrep = r_mem.get_final_state()
#            r_enc = r_mem.get_initial_input_enc()
#            self.r_mrep = tf.reshape(self.r_mrep, [batch_size, -1, mem_dim])

    def _encode_p(self, scope_name):
        if not self.use_concat_p:
            pass
        else:
            with tf.variable_scope('{}_encoding'.format(scope_name), reuse = tf.AUTO_REUSE):
                # sep_p_encodes = [batch_size, max_length, fw_hideen+bw_hideen=2*hidden]
                # sep_p_final_state = [batch_size, fw_hidden+bw_hidden=2*hidden]
                self.sep_p_encodes, self.sep_p_final_state = rnn(self.rnn_type, self.p_emb, self.p_length, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
            if self.use_char_embedding:
                with tf.variable_scope('{}_char_encoding'.format(score_name), reuse = tf.AUTO_REUSE):
                    # sep_p_encodes_char is already concated  by fw_state and bw_state
                    # [batch_size , 2*hidden]
                    _, sep_p_encodes_char = rnn(self.rnn_type, self.p_emb_char, self.p_t_length, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
                    # self.p is [batch_size, max_length]
                    # [batch_size * max_length, 2*hidden] can be reshape into [batch_size, max_length, 2*hidden]
                    # after reshaping, then every last state is just aligned to corresponding word embeding.self.sep_p_encodes_char = tf.reshape(sep_p_encodes_char, [-1, tf.shape(self.p)[1], 2 * self.hidden_size])
                    # concat last char last(just like a word_emebeding) to the outputs of the every word
                    # [batch_size, max_len, 4 * hidden]
                    self.sep_p_encodes = tf.concat([self.sep_p_encodes, self.sep_p_encodes_char], 2)
            if self.use_dropout:
                self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
                # self.sep_p_final_state = tf.nn.dropout(self.sep_p_final_state, self.dropout_keep_prob)

    def _encode_r(self, scope_name):
        if not self.use_response_u:
            pass
        else:
            with tf.variable_scope('{}_encoding'.format(scope_name), reuse=tf.AUTO_REUSE):
                # for response, we don't use the hierachy
                # sep_r_encodes = [batch_size, max_length, fw_hideen+bw_hideen=2*hidden]
                # sep_r_final_state = [batch_size, fw_hideen+bw_hideen=2*hidden]
                self.sep_r_encodes, self.sep_r_final_state = rnn(self.rnn_type, self.r_emb, self.r_length, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
                if self.use_char_embedding:
                    with tf.variable_scope('{}_char_encoding'.format(scope_name), reuse=tf.AUTO_REUSE):
                        # sep_r_encodes_char is already concated  by fw_state and bw_state
                        # [batch_size*max_len, 2*hidden]
                        _, sep_r_encodes_char = rnn(self.rnn_type, self.r_emb_char, self.r_t_length, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
                        # reshaping into [batch_size, max_len, 2*hidden]
                        self.sep_r_encodes_char = tf.reshape(sep_r_encodes_char, [-1, tf.shape(self.r)[1], 2 * self.hidden_size])
                        # [batch_size, max_len, 4 * hidden]
                        self.sep_r_encodes = tf.concat([self.sep_r_encodes, self.sep_r_encodes_char], 2)

                if self.use_dropout:
                    self.sep_r_encodes = tf.nn.dropout(self.sep_r_encodes, self.dropout_keep_prob)

    def _encode_q(self, scope_name):
        if not self.use_question:
            pass
        else:
            with tf.variable_scope('{}_encoding'.format(scope_name), reuse = tf.AUTO_REUSE):
                # for question, we don't use the hierachy first.
                # sep_q_encodes = [batch_size, max_length, fw_hideen+bw_hideen=2*hidden]
                # sep_q_final_state = [batch_size, fw_hideen+bw_hideen=2*hidden]
                self.sep_q_encodes, self.sep_q_final_state = rnn(self.rnn_type, self.q_emb, self.q_length, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
                if self.use_char_embedding:
                    with tf.variable_scope('{}_char_encoding'.format(scope_name), reuse = tf.AUTO_RESUE):
                        # sep_q_encodes_char is already concated  by fw_state and bw_state
                        # [batch_size*max_len, 2*hidden]
                        _, sep_q_encodes_char = rnn(self.rnn_type, self.q_emb_char, self.q_t_length, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
                        # reshaping into [batch_size, max_len, 2*hidden]
                        self.sep_q_encodes_char = tf.reshape(sep_q_encodes_char, [-1, tf.shape(self.q)[1], 2 * self.hidden_size])
                        # [batch_size, max_len, 4 * hidden]
                    self.sep_q_encodes = tf.concat([self.sep_q_encodes, self.sep_q_encodes_char], 2)

                if self.use_dropout:
                    self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _hierarchy_match(self):
        """
        # Jie
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        elif self.algo == 'GMLSTM' or self.algo == 'RNET':
            match_layer = GatedMatchLSTMLayer(self.hidden_size)
        elif self.algo == 'GA':
            match_layer = GatedAttentionLayer(self.hidden_size)
        elif self.algo == 'CGA':
            match_layer = CollaborativeGatedAttentionLayer(self.hidden_size)
        elif self.algo == 'CGMLSTM':
            match_layer = CollaborativeGatedMatchLSTMLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))

        if self.hierarchy_q_pu_att and self.use_question:
            with tf.variable_scope('question-aware-pu'):
                # make the shape
                # sep_p_u_ecnodes = [batch_size * max_passage_window, max_u_len,2 * hidden_size]
                # sep_q_encodes = [batch_size, max_q_len, 2* hidden_size]
                self.dup_sep_q_for_pu_encodes = tf_utils._tf_dup_3(self.sep_q_encodes, self.max_passage_window, self.encode_output_size)
                self.match_q_pu_encodes, _ = match_layer.match(
                    self.sep_p_u_encodes, self.dup_sep_q_for_pu_encodes,
                    self.p_u_length, self.dup_q_for_pu_length, self.p_utterance_mask, self.dup_q_for_pu_mask
                )
                if self.use_dropout:
                    self.match_q_pu_encodes = tf.nn.dropout(self.match_q_pu_encodes, self.dropout_keep_prob)

        if self.hierarchy_r_pu_att and self.use_response_u:
            with tf.variable_scope('response-aware-pu'):
                # make the shape
                # sep_p_u_ecnodes = [batch_size * max_passage_window, max_u_len,2 * hidden_size]
                # if for use_r_in_seq_word_match
                # sep_p_u_ecnodes = [batch_size * (max_passage_window+1), max_u_len,2 * hidden_size]
                if self.use_r_in_wm_seq:
                    self.max_wm_seq_length = self.max_passage_window + 1
                    # sep_p_u_encodes= [batch_size * max_passage_window, max_u_length, fw_hidden+bw_hidden=2*hidden]
                    reshaped_p_u_encodes = tf.reshape(self.sep_p_u_encodes, [tf.shape(self.p_wsize)[0], -1, self.max_u_len, self.encode_output_size])
                    # sep_r_encodes = [batch_size * 1, max_u_len, 2* hidden_size]
                    reshaped_r_u_encodes = tf.expand_dims(self.sep_r_encodes, axis = 1)
                    self.wm_seq_encodes = tf.reshape(tf.concat([reshaped_p_u_encodes, reshaped_r_u_encodes], 1), [-1,self.max_u_len, self.encode_output_size])
                    # p_u_length = [batch_size * max_passage_window]
                    # reshaped_p_u_length = [batch_size, max_passage_window]
                    reshaped_p_u_length = tf.reshape(self.p_u_length,[-1, self.max_passage_window])
                    # r_length = [batch_size]
                    # reshapred_r_length = [batch_size, 1]
                    reshaped_r_length = tf.expand_dims(self.r_length, axis = 1)
                    # wm_seq_length = [batch_size * (max_passage_window+1)]
                    self.wm_seq_length = tf.reshape(tf.concat([reshaped_p_u_length, reshaped_r_length], -1), [-1])
                    # [batch_size * (max_passage_window_size+1), max_u_length]
                    self.wm_seq_mask = tf.sequence_mask(self.wm_seq_length, tf.shape(self.p_u)[-1], dtype=tf.float32)
                else:
                    self.max_wm_seq_length = self.max_passage_window
                    self.wm_seq_length = self.p_u_length
                    self.wm_seq_encodes = self.sep_p_u_encodes
                    self.wm_seq_mask = self.p_utterance_mask

                # sep_r_encodes = [batch_size, max_u_len, 2* hidden_size]
                self.dup_sep_r_for_wm_seq_encodes = tf_utils._tf_dup_3(self.sep_r_encodes, self.max_wm_seq_length, self.encode_output_size)
                # dup_r_for_word_match_
                # [batch_size * self.max_passage_window]
                self.dup_r_for_wm_seq_length = tf_utils._tf_dup_1(self.r_length, self.max_wm_seq_length)
                # [batch_size * max_wm_seq_length, max_length]
                self.dup_r_for_wm_seq_mask = tf.sequence_mask(self.dup_r_for_wm_seq_length, tf.shape(self.r)[-1], dtype=tf.float32)
                self.match_r_pu_encodes, _ = match_layer.match(
                    self.wm_seq_encodes, self.dup_sep_r_for_wm_seq_encodes,
                    self.wm_seq_length, self.dup_r_for_wm_seq_length, self.wm_seq_mask, self.dup_r_for_wm_seq_mask
                )
                if self.use_dropout:
                    self.match_r_pu_encodes = tf.nn.dropout(self.match_r_pu_encodes, self.dropout_keep_prob)


    def _high_encode(self):
        """
        # Jie
        Used for hierarchy lstem, feed pu into a higher sentence-level lstm
        """
        # put every utterance final state into a sentence-level lstm
        # [batch_size, max_passage_window, fw_hidden + bw_hidden = 2*hidden]
        if self.hierarchy_q_pu_att and self.use_question:
            # [batch, window_size, hidden_size]
            self.reshaped_p_utterance_state = tf.reshape(self.fuse_q_pu_final_state, [tf.shape(self.p_wsize)[0], -1, 2 * self.hidden_size])
        elif self.hierarchy_r_pu_att and self.use_response_u:
            self.reshaped_fuse_r_pu_final_state = tf.reshape(self.fuse_r_pu_final_state, [tf.shape(self.p_wsize)[0], -1, 2 * self.hidden_size])
            if self.use_r_in_wm_seq:
                # put every utterance final state into a sentence-level lstm
                # [batch_size, max_passage_window+1, fw_hidden + bw_hidden = 2*hidden]
                # We haved do preceding adding to make exactly max_passage_window utterances
                self.reshaped_p_utterance_state = self.reshaped_fuse_r_pu_final_state[:,:-1,:]
            else:
                # put every utterance final state into a sentence-level lstm
                # [batch_size, max_passage_window, fw_hidden + bw_hidden = 2*hidden]
                self.reshaped_p_utterance_state = self.reshaped_fuse_r_pu_final_state
        else:
            self.reshaped_p_utterance_state = tf.reshape(self.sep_p_u_final_state, [tf.shape(self.p_wsize)[0], -1, 2 * self.hidden_size])

        if self.use_dropout:
            # usually we don't dropout final state, while in the hierarchy lstm, we dropout the lower final state
            # [batch, window_size, hidden_size]
            self.reshaped_p_utterance_state = tf.nn.dropout(self.reshaped_p_utterance_state, self.dropout_keep_prob)

        # it is the concatnated q
        if self.use_question:
            q_final_state = tf.expand_dims(self.sep_q_final_state,1)
            if self.use_dropout:
                # [batch, window_size, hidden_size]
                q_final_state = tf.nn.dropout(q_final_state, self.dropout_keep_prob)

        if self.use_response_u:
            if self.hierarchy_r_pu_att and self.use_response_u and self.use_r_in_wm_seq:
                # split out the attented r as final state
                r_final_state = self.reshaped_fuse_r_pu_final_state[:,-1,:]
            else:
                # original r state, without attention
                r_final_state = self.sep_r_final_state

            if self.use_dropout:
                r_final_state = tf.nn.dropout(r_final_state, self.dropout_keep_prob)

        self.snt_size = self.p_wsize
        self.snt_mask = self.p_window_mask

        # We always use the biGRU in the word-lvl. TODO: support gru and other vector dimension
        dim_total = 2 * self.hidden_size

        # add speaker embedding
        if self.use_speaker_embedding:
            if self.snt_att_algo in ['snt_self_att', 'snt_qu_pu_att', 'snt_r_pu_att','']:
                self.reshaped_p_u_speaker_emb = tf.reshape(self.p_u_speaker_emb, [tf.shape(self.p_wsize)[0], -1, self.speaker_embed_dim])
                self.reshaped_p_utterance_state = tf.concat([self.reshaped_p_utterance_state, self.reshaped_p_u_speaker_emb], -1)
                dim_total = dim_total + self.speaker_embed_dim
                if self.use_response_u:
                    r_final_state = tf.concat([r_final_state, self.r_speaker_emb], -1)
                    if self.use_r_in_seq:
                        # add r after pu seqs
                        # batch_size, qu_length+ 1, dim
                        self.reshaped_p_utterance_state = tf.concat([self.reshaped_p_utterance_state, tf.expand_dims(r_final_state, 1)], -2)
                        self.snt_size = self.p_wsize + 1
                        self.snt_mask = tf.sequence_mask(self.snt_size, self.max_passage_window+1, dtype=tf.float32)
        else:
            # if we need add r in
            if self.use_r_in_seq:
                # add r after pu seqs
                # batch_size, qu_length+ 1, dim
                self.reshaped_p_utterance_state = tf.concat([self.reshaped_p_utterance_state, tf.expand_dims(r_final_state, 1)], -2)
                self.snt_size = self.p_wsize + 1
                self.snt_mask = tf.sequence_mask(self.snt_size, self.max_passage_window+1, dtype=tf.float32)

        # for r_pu_att, we cannot add r with speaker and label embedding all the tiem, because we don't have the label for r.
        # add label embedding
        if self.use_label_embedding:
            # not use r label
            if not self.use_r_in_seq and self.snt_att_algo in ['snt_self_att', 'snt_qu_pu_att', '']:
                # batch_size, pu_length, label_embedding_dim
                self.reshaped_p_u_label_emb = tf.reshape(self.p_u_label_emb, [tf.shape(self.p_wsize)[0], -1, self.label_embed_dim])
                self.reshaped_p_utterance_state = tf.concat([self.reshaped_p_utterance_state, self.reshaped_p_u_label_emb], -1)
                # batch_size, qu_length, label_embedding_dim
                dim_total = dim_total + self.label_embed_dim

        if self.use_snt_att:
            if self.snt_att_algo == 'snt_self_att':
                self.reshaped_p_utterance = self._multi_hops_snt_seq_encode(self.reshaped_p_utterance_state, dim_total, self.snt_seq_hops)
            elif self.snt_att_algo == 'snt_r_pu_att' and self.use_response_u:
                # [batch_size, q_size, k_size]
                # [batch_size, pu_window, 1]
                snt_r_pu_mask = tf.expand_dims(self.snt_mask, axis=2)
                r_final_state = tf.expand_dims(r_final_state, 1)
                self.reshaped_p_utterance = multihead_attention(dim_total, self.num_att_head, self.reshaped_p_utterance_state, memory = r_final_state, mask = snt_r_pu_mask, scope = 'snt_r_pu_att')
            elif self.snt_att_algo == 'snt_q_pu_att' and self.use_question:
                # [batch_size, q_size, k_size]
                # [batch_size, pu_window , question_window]
                snt_q_pu_mask = tf.expand_dims(self.snt_mask, axis=2)
                self.reshaped_p_utterance = multihead_attention(dim_total, self.num_att_head, self.reshaped_p_utterance_state, memory = q_final_state, mask = snt_q_pu_mask, scope = 'snt_q_pu_att')
            elif self.snt_att_algo == 'snt_qu_pu_att':
                snt_qu_pu_mask = tf.tile(tf.expand_dims(self.snt_mask, axis=2), [1,1,self.question_window])
                qu_state = self.reshaped_p_utterance_state[:,:-self.question_window,:]
                self.reshaped_p_utterance = multihead_attention(dim_total, self.num_att_head, self.reshaped_p_utterance_state, memory = qu_state, mask = snt_qu_pu_mask, scope = 'snt_qu_pu_att')
            else:
                raise NotImplementedError('The snt-att algorithm {} is not implemented.'.format(self.snt_att_algo))

        # use a final snt-level rnn to organize the context utterances.
        with tf.variable_scope('hierarchy_passage_encoding'):
            if self.decode_inputs == ['label_only']:
                self.sep_hp_encodes, self.sep_hp_final_state = rnn(self.snt_rnn_type, self.reshaped_p_u_label_emb, self.snt_size, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
            elif self.decode_inputs == ['sp_label_only']:
                self.sep_hp_encodes, self.sep_hp_final_state = rnn(self.snt_rnn_type, tf.concat([self.reshaped_p_u_label_emb, self.reshaped_p_u_speaker_emb], -1), self.snt_size, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
            else:
                # [batch_size, max_passage_window, fw_hidden + bw_hidden = 2*hidden]
                self.sep_hp_encodes, self.sep_hp_final_state = rnn(self.snt_rnn_type, self.reshaped_p_utterance_state, self.snt_size, self.hidden_size, dropout_keep_prob = self.dropout_keep_prob)
        if self.use_dropout:
            self.sep_hp_encodes = tf.nn.dropout(self.sep_hp_encodes, self.dropout_keep_prob)
            # self.sep_hp_final_state = tf.nn.dropout(self.sep_hp_final_state, self.dropout_keep_prob)


    def _multi_hops_snt_seq_encode(self, snt_encodes, dim_total, hops):
        for i in range(self.snt_seq_hops):
            snt_seq_encoding_scope = 'snt_seq_encoding_%s' % i
            snt_seq_selfatt_scope = 'snt_seq_selfatt_%s' % i
            hidden_size = self.hidden_size
            snt_encodes = self._snt_seq_encode(snt_encodes, self.hidden_size, snt_seq_encoding_scope)
            dim_total = hidden_size
            snt_encodes = self._snt_seq_selfatt(snt_encodes, dim_total, snt_seq_selfatt_scope)
        return snt_encodes

    def _snt_seq_encode(self, snt_encodes, hidden_size, scope='snt_sub_encoding'):
        # use a final snt-level rnn to organize the context utterances.
        with tf.variable_scope(scope):
            # [batch_size, max_passage_window, fw_hidden + bw_hidden = 2*hidden]
            snt_encodes, _ = rnn(self.snt_rnn_type, snt_encodes, self.snt_size, hidden_size, dropout_keep_prob = self.dropout_keep_prob)
        if self.use_dropout:
            snt_encodes = tf.nn.dropout(snt_encodes, self.dropout_keep_prob)
        return snt_encodes

    def _snt_seq_selfatt(self, snt_encodes, dim_total, scope = 'snt_self_att'):
        num_heads = self.num_att_head
        # query [batch_size, max_passage_window, hidden]
        # [batch_size, max_passage_window]
        max_snt_size = tf.shape(self.snt_mask)[1]
        # [batch_size, snt_mask, snt_mask]
        snt_window_mask = tf.tile(tf.expand_dims(self.snt_mask, axis=1), [1, max_snt_size, 1])
        # [batch_size, max_passage_window, hidden]
        out = multihead_attention(dim_total, num_heads, snt_encodes, snt_encodes, snt_window_mask, scope)
        return out

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        elif self.algo == 'GMLSTM' or self.algo == 'RNET':
            match_layer = GatedMatchLSTMLayer(self.hidden_size)
        elif self.algo == 'GA':
            match_layer = GatedAttentionLayer(self.hidden_size)
        elif self.algo == 'CGA':
            match_layer = CollaborativeGatedAttentionLayer(self.hidden_size)
        elif self.algo == 'CGMLSTM':
            match_layer = CollaborativeGatedMatchLSTMLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))

        if self.flat_c_q_att and self.use_concat_p and self.use_question:
            with tf.variable_scope('context-aware-q'):
                self.match_q_encodes, _ = match_layer.match(
                    self.sep_q_encodes, self.sep_p_encodes,
                    self.q_length, self.p_length, self.q_mask, self.p_mask
                )
                if self.use_dropout:
                    self.match_q_encodes = tf.nn.dropout(self.match_q_encodes, self.dropout_keep_prob)

        if self.flat_q_c_att and self.use_concat_p and self.use_question:
            with tf.variable_scope('question-aware-c'):
                self.match_qc_encodes, _ = match_layer.match(
                    self.sep_p_encodes, self.sep_q_encodes,
                    self.p_length, self.q_length, self.p_mask, self.q_mask
                )
                if self.use_dropout:
                    self.match_qc_encodes = tf.nn.dropout(self.match_qc_encodes, self.dropout_keep_prob)

        if self.flat_c_r_att and self.use_concat_p and self.use_response_u:
            with tf.variable_scope('context-aware-r'):
                self.match_r_encodes, _ = match_layer.match(
                    self.sep_r_encodes, self.sep_p_encodes,
                    self.r_length, self.p_length, self.r_mask, self.p_mask
                )
                if self.use_dropout:
                    self.match_r_encodes = tf.nn.dropout(self.match_r_encodes, self.dropout_keep_prob)

        if self.flat_r_c_att and self.use_concat_p and self.use_response_u:
            with tf.variable_scope('response-aware-c'):
                self.match_rc_encodes, _ = match_layer.match(
                    self.sep_p_encodes, self.sep_r_encodes,
                    self.p_length, self.r_length, self.p_mask, self.r_mask
                )
                if self.use_dropout:
                    self.match_rc_encodes = tf.nn.dropout(self.match_rc_encodes, self.dropout_keep_prob)


    # self-attention layer for hierarchy q_pu
    def _hierarchy_selfatt(self):
        """
        Self-attention on the question-aware passage encoding
        """
        match_layer = GatedMatchLSTMLayer(self.hidden_size)

        # firstly run through another biLSTM on match_p_encodes. Then apply self-attention
        if self.hierarchy_q_pu_att and self.use_question:
            with tf.variable_scope('selfatt_q_pu'):
                self.match_q_pu_encodes, _ = match_layer.match(
                    self.match_q_pu_encodes, self.match_q_pu_encodes,
                    self.p_u_length, self.p_u_length, self.p_utterance_mask, self.p_utterance_mask
                )
                if self.use_dropout:
                    self.match_q_pu_encodes = tf.nn.dropout(self.match_q_pu_encodes, self.dropout_keep_prob)

        if self.hierarchy_r_pu_att and self.use_response_u:
            with tf.variable_scope('selfatt_r_pu'):
                self.match_r_pu_encodes, _ = match_layer.match(
                    self.match_r_pu_encodes, self.match_r_pu_encodes,
                    self.p_u_length, self.p_u_length, self.p_utterance_mask, self.p_utterance_mask
                )
                if s_lf.use_dropout:
                    self.match_r_pu_encodes = tf.nn.dropout(self.match_r_pu_encodes, self.dropout_keep_prob)



    # self-attention layer like in R-net
    def _selfatt(self):
        """
        Self-attention on the question-aware passage encoding
        """
        match_layer = GatedMatchLSTMLayer(self.hidden_size)

        # firstly run through another biLSTM on match_p_encodes. Then apply self-attention
        if self.flat_c_q_att and self.use_concat_p and self.use_question:
            with tf.variable_scope('selfatt_q'):
                self.match_q_encodes, _ = match_layer.match(
                    self.match_q_encodes, self.match_q_encodes,
                    self.q_length, self.q_length, self.q_mask, self.q_mask
                )
                if self.use_dropout:
                    self.match_q_encodes = tf.nn.dropout(self.match_q_encodes, self.dropout_keep_prob)

        if self.flat_q_c_att and self.use_concat_p and self.use_question:
            with tf.variable_scope('selfatt_qc'):
                self.match_qc_encodes, _ = match_layer.match(
                    self.match_qc_encodes, self.match_qc_encodes,
                    self.p_length, self.p_length, self.p_mask, self.p_mask
                )
                if self.use_dropout:
                    self.match_qc_encodes = tf.nn.dropout(self.match_qc_encodes, self.dropout_keep_prob)
        # firstly run through another biLSTM on match_p_encodes. Then apply self-attention
        if self.flat_c_r_att and self.use_concat_p and self.use_response_u:
            with tf.variable_scope('selfatt_r'):
                self.match_r_encodes, _ = match_layer.match(
                    self.match_r_encodes, self.match_r_encodes,
                    self.r_length, self.r_length, self.r_mask, self.r_mask
                )
                if self.use_dropout:
                    self.match_r_encodes = tf.nn.dropout(self.match_r_encodes, self.dropout_keep_prob)

        if self.flat_r_c_att and self.use_concat_p and self.use_response_u:
            with tf.variable_scope('selfatt_rc'):
                self.match_rc_encodes, _ = match_layer.match(
                    self.match_rc_encodes, self.match_rc_encodes,
                    self.p_length, self.p_length, self.p_mask, self.p_mask
                )
                if self.use_dropout:
                    self.match_rc_encodes = tf.nn.dropout(self.match_rc_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        if self.flat_c_q_att and self.use_concat_p and self.use_question:
            with tf.variable_scope('fusion_cq'):
                # TODO: fuse always bi-lstm?
                # The shape of self.fuse_q_encodes is[BatchSize, MaxPassageLength, hidden_size*2]
                self.fuse_cq_encodes, self.fuse_cq_final_state = rnn(self.rnn_type, self.match_q_encodes, self.q_length, self.hidden_size, layer_num=1, dropout_keep_prob = self.dropout_keep_prob)
                if self.use_dropout:
                    self.fuse_cq_encodes = tf.nn.dropout(self.fuse_cq_encodes, self.dropout_keep_prob)

        if self.flat_q_c_att and self.use_concat_p and self.use_question:
            with tf.variable_scope('fusion_qc'):
                # The shape of self.fuse_qc_encodes is[BatchSize, MaxPassageLength, hidden_size*2]
                self.fuse_qc_encodes, self.fuse_qc_final_state = rnn(
                    self.rnn_type, self.match_qc_encodes, self.p_length,
                    self.hidden_size, layer_num=1,
                    dropout_keep_prob = self.dropout_keep_prob
                )
                if self.use_dropout:
                    self.fuse_qc_encodes = tf.nn.dropout(self.fuse_qc_encodes, self.dropout_keep_prob)

        if self.flat_c_r_att and self.use_concat_p and self.use_response_u:
            with tf.variable_scope('fusion_cr'):
                self.fuse_cr_encodes, self.fuse_cr_final_state = rnn(self.rnn_type, self.match_r_encodes, self.r_length, self.hidden_size, layer_num=1, dropout_keep_prob = self.dropout_keep_prob)
                if self.use_dropout:
                    self.fuse_cr_encodes = tf.nn.dropout(self.fuse_cr_encodes, self.dropout_keep_prob)

        if self.flat_r_c_att and self.use_concat_p and self.use_response_u:
            with tf.variable_scope('fusion_rc'):
                # The shape of self.fuse_qc_encodes is[BatchSize, MaxPassageLength, hidden_size*2]
                self.fuse_rc_encodes, self.fuse_rc_final_state = rnn(
                    self.rnn_type, self.match_rc_encodes, self.p_length,
                    self.hidden_size, layer_num=1, dropout_keep_prob = self.dropout_keep_prob)
                if self.use_dropout:
                    self.fuse_rc_encodes = tf.nn.dropout(self.fuse_rc_encodes, self.dropout_keep_prob)



    def _hierarchy_fuse(self):
        """
        when hierarchy_q_pu_att is true, it means we use question aware pu, after matching, we usually use the fuse layer to get the final encoding after matching.
        """
        if self.hierarchy_q_pu_att and self.use_question:
            with tf.variable_scope('fusion_q_pu'):
                # The shape of self.fuse_q_pu_encodes is [batchSize * max_passage_window, max_u_len, hidden_size*2]
                # as fuse is optional, just make sep_p_u_final_state to receive the fuse state
                self.fuse_q_pu_encodes, self.fuse_q_pu_final_state = rnn(
                    self.rnn_type, self.match_q_pu_encodes, self.p_u_length,
                    self.hidden_size, layer_num=1, dropout_keep_prob = self.dropout_keep_prob)
                if self.use_dropout:
                    self.fuse_q_pu_encodes = tf.nn.dropout(self.fuse_q_pu_encodes, self.dropout_keep_prob)
                    # self.fuse_q_pu_final_state = tf.nn.dropout(self.fuse_q_pu_final_state, self.dropout_keep_prob)

        if self.hierarchy_r_pu_att and self.use_response_u:
            with tf.variable_scope('fusion_r_pu'):
                # The shape of self.fuse_q_pu_encodes is [batchSize * max_passage_window, max_u_len, hidden_size*2]
                # as fuse is optional, just make sep_p_u_final_state to receive the fuse state
                self.fuse_r_pu_encodes, self.fuse_r_pu_final_state = rnn(
                    self.rnn_type, self.match_r_pu_encodes, self.wm_seq_length,
                    self.hidden_size, layer_num=1,dropout_keep_prob = self.dropout_keep_prob
                )
                if self.use_dropout:
                    self.fuse_r_pu_encodes = tf.nn.dropout(self.fuse_r_pu_encodes, self.dropout_keep_prob)
                    # self.fuse_q_pu_final_state = tf.nn.dropout(self.fuse_q_pu_final_state, self.dropout_keep_prob)

    def _conv_max_pooling(self, name, encodes, max_len_of_encodes):
        """
        for  reference, http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
        """
        encodes_expanded = tf.expand_dims(encodes, -1)
        encodes_expanded_pad = tf.pad(
            encodes_expanded,
            [
                [0, 0],
                [0, max_len_of_encodes - tf.shape(encodes_expanded)[1]],
                [0, 0],
                [0, 0]
            ]
        )
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            # print 'filter_size', filter_size
            with tf.name_scope("conv-maxpool-%s-%s" % (name, filter_size)):
                # Convolution Layer
                filter_shape = [filter_size, self.state_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    encodes_expanded_pad,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_len_of_encodes - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
        return h_pool_flat

    def _decode_func_multiclass(self, decode_func, num_classes, inputs_rep, input_dim, name):
        if decode_func == 'WX_B':
            W = tf.get_variable(
                "W_{}_final".format(name),
                shape=[input_dim, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], name="b{}".format(name)))
            scores = tf.reshape(tf.nn.xw_plus_b(inputs_rep, W, b), [-1, num_classes])
        elif decode_func == 'FC':
            # when no action, it is just the WX_B
            # scores = tf.contrib.layers.fully_connected(inputs = inputs_rep, num_outputs=num_classes, activation_fn=None)
            # do not use relu before the final layer, which will cut all the negative value
            # just the input_dim/2 as the dim of hidden fully connected_layers
            # mid_layers = tf.contrib.layers.fully_connected(inputs = inputs_rep, num_outputs=64)
            if self.use_batch_norm:
                h1 = tf.contrib.layers.fully_connected(inputs = inputs_rep, num_outputs=input_dim/2, activation_fn=None,scope='linear_dense')
                mid_layers = tf.layers.batch_normalization(h1, center=True, scale = False, training=self.is_training_phase)
                mid_layers = tf.nn.relu(mid_layers, 'relu')
            else:
                mid_layers = tf.contrib.layers.fully_connected(inputs = inputs_rep, num_outputs=input_dim/2)

            # also add dropout after the nolinear layer before output layer
            if self.use_dropout_mlp:
                mid_layers = tf.nn.dropout(mid_layers, self.dropout_keep_prob_mlp)

            scores = tf.contrib.layers.fully_connected(inputs = mid_layers, num_outputs=num_classes, activation_fn=None)
            #scores = tf.contrib.layers.fully_connected(inputs = inputs_rep, num_outputs=num_classes, activation_fn=lambda x : tf.nn.leaky_relu(x, alpha = 0.01))
        elif decode_func == 'BILINEAR':
            mid_layers = tf.contrib.layers.fully_connected(inputs = inputs_rep, num_outputs=input_dim/2)
            # also add dropout after the nolinear layer before output layer
            if self.use_dropout_mlp:
                mid_layers = tf.nn.dropout(mid_layers, self.dropout_keep_prob_mlp)

            generated_response = tf.contrib.layers.fully_connected(inputs = mid_layers, num_outputs=self.label_embed_dim, activation_fn=None)
            generated_response = tf.expand_dims(generated_response, 1)
            # label embedding is [self.numclasses, self.label_embed_dim]
            # after reshape is [batch_size, self.label_embed_dim, self.num_classes]
            tiled_pred_label_encoding = tf.transpose(tf.tile(tf.expand_dims(self.pred_label_emb, 0), [tf.shape(inputs_rep)[0],1,1]), perm=[0,2,1])
            # transpose last two dimensions
            # Dot product between generated response and actual response to calculate the cos similarties
            # (c * M) * r
            # self.similarities = [batch_size, 1, self.num_classes]
            label_sims = tf.matmul(generated_response, tiled_pred_label_encoding)
            # [batch_size, self.max_candidate_answers]
            scores = tf.squeeze(label_sims, [1])
        else:
            raise NotImplemented("decode func {} is not implemented yet.", decode_func)
        return scores


    def _decode_with_final_state(self):
        self.scores = 0.0
        self.final_states = []
        self.final_dim = 0

        if 'label_only' in self.decode_inputs or 'sp_label_only' in self.decode_inputs:
            if self.dial_encode == 'HIERARCHY':
                # [batch_size, max_passage_window, fw_hidden + bw_hidden = 2*hidden]
                l_final_state = self.sep_hp_final_state
            else:
                raise NotImplementedError('only HIERARCHY is supported for label_only and sp_label_only')

            l_final_dim = self.state_size
            with tf.name_scope("decode_l_final"):
                self.final_states.append(l_final_state)
                self.final_dim += l_final_dim
                if self.decode_combine == 'additive':
                    self.scores += self._decode_func_multiclass(self.decode_func, self.num_classes, l_final_state, l_final_dim, "l")

        # default value for a_final_state
        if 'p_final' in self.decode_inputs:
            # concat then rnn
            # default value for p_final_state
            if self.use_concat_p:
                p_final_state = self.sep_p_final_state
            # we use hp_final_state when using hierarchy
            if self.algo == 'MEMNET':
                p_final_state = self.p_u_mrep
            else:
                if self.dial_encode == 'HIERARCHY':
                    # [batch_size, max_passage_window, fw_hidden + bw_hidden = 2*hidden]
                    p_final_state = self.sep_hp_final_state

            p_final_dim = self.state_size
            with tf.name_scope("decode_p_final"):
                p_final_state, p_final_dim = self._decode_extra_features(p_final_state, p_final_dim)
                self.final_states.append(p_final_state)
                self.final_dim += p_final_dim
                if self.decode_combine == 'additive':
                    self.scores += self._decode_func_multiclass(self.decode_func, self.num_classes, p_final_state, p_final_dim, "p")

        if 'q_final' in self.decode_inputs and self.use_question:
            # q rep
            q_final_state = self.sep_q_final_state
            if self.flat_c_q_att:
                q_final_state = self.fuse_cq_final_state

            q_final_dim = self.state_size
            with tf.name_scope("decode_q_final"):
                q_final_state, q_final_dim = self._decode_extra_features(q_final_state, q_final_dim)
                self.final_states.append(q_final_state)
                self.final_dim += q_final_dim
                if self.decode_combine == 'additive':
                    self.scores += self._decode_func_multiclass(self.decode_func, self.num_classes, q_final_state, q_final_dim, "q")

        if 'qu_final' in self.decode_inputs and self.dial_encode == 'HIERARCHY':
            qu_final_dim = self.state_size
            # [batch_size, time, state_size]
            qu_final_state = self.sep_hp_encodes[:,-1,:]
            with tf.name_scope("decode_qu_final"):
                qu_final_state, qu_final_dim = self._decode_extra_features(qu_final_state, qu_final_dim)
                self.final_states.append(qu_final_state)
                self.final_dim += qu_final_dim
                if self.decode_combine == 'additive':
                    self.scores += self._decode_func_multiclass(self.decode_func, self.num_classes, qu_final_state, qu_final_dim, "qu")

        if 'ru_final' in self.decode_inputs and self.use_r_in_seq and self.dial_encode == 'HIERARCHY':
            ru_final_dim = self.state_size
            # [batch_size, time, state_size]
            ru_final_state = self.sep_hp_encodes[:,-1,:]
            with tf.name_scope("decode_ru_final"):
                ru_final_state, ru_final_dim = self._decode_extra_features(ru_final_state, ru_final_dim)
                self.final_states.append(ru_final_state)
                self.final_dim += ru_final_dim
                if self.decode_combine == 'additive':
                    self.scores += self._decode_func_multiclass(self.decode_func, self.num_classes, ru_final_state, ru_final_dim, "ru")

        if 'r_final' in self.decode_inputs and self.use_response_u:
            # r rep
            r_final_state = self.sep_r_final_state
            if self.algo == 'MEMNET':
                r_final_state = self.r_mrep
            if self.flat_c_r_att:
                r_final_state = self.fuse_cr_final_state

            r_final_dim = self.state_size
            with tf.name_scope("decode_r_final"):
                r_final_state, r_final_dim = self._decode_extra_features(r_final_state, r_final_dim)
                self.final_states.append(r_final_state)
                self.final_dim += r_final_dim
                if self.decode_combine == 'additive':
                    self.scores += self._decode_func_multiclass(self.decode_func, self.num_classes, r_final_state, r_final_dim, "r")


        # for decode_combine == concat, we concat all , then go through the fc layer.
        if self.decode_combine == 'concat':
            self.scores += self._decode_func_multiclass(self.decode_func, self.num_classes, tf.concat(self.final_states, -1), self.final_dim, "concat_all")

    # final_state is a tensor, which is passby value of the reference, hence the content in that referenced object be changed in this function
    # final_dim is int, which is passing by value, it fill not be changed.
    def _decode_extra_features(self, final_state, final_dim):
        if self.use_speaker_embedding and self.use_response_s:
            final_state = tf.concat([final_state, self.r_speaker_emb], -1)
            final_dim = final_dim + self.speaker_embed_dim

        # for now, only use the last label for decoding.
        if self.use_label_embedding and self.decode_r_with_flatten_pu_labels:
            self.flatten_decode_label_emb = tf.reshape(self.p_u_label_emb, [tf.shape(self.p_wsize)[0], self.label_embed_dim * self.context_window])
            final_state = tf.concat([final_state, self.flatten_decode_label_emb], -1)
            final_dim = final_dim + self.label_embed_dim * self.context_window

        return final_state, final_dim


    def _decode_with_cnn_pooling(self):
        """
        concat all features.
        """
        concat_arr = []

        if self.flat_c_q_att and self.use_concat_p and self.use_question:
            c_q_att_fuse_final = self._conv_max_pooling("cq", self.fuse_q_encodes, self.max_q_len)
            concat_arr.append(c_q_att_fuse_final)

        if self.flat_q_c_att and self.use_concat_p and self.use_question:
            q_c_fuse_final = self._conv_max_pooling("qc", self.fuse_qc_encodes, self.max_p_len)
            concat_arr.append(q_c_att_fuse_final)

        if self.flat_c_r_att and self.use_concat_p and self.use_response_u:
            c_r_att_fuse_final = self._conv_max_pooling("cr", self.fuse_r_encodes, self.max_u_len)
            concat_arr.append(c_r_att_fuse_final)

        if self.flat_r_c_att and self.use_concat_p and self.use_response_u:
            r_c_fuse_final = self._conv_max_pooling("rc", self.fuse_rc_encodes, self.max_p_len)
            concat_arr.append(r_c_att_fuse_final)

        if self.decode_p and self.use_concat_p:
            # decode_p_final [batch_size, max_len, number_total_filters]
            decode_p_final = self._conv_max_pooling("p", self.sep_p_encodes, self.max_p_len)
            concat_arr.append(decode_p_final)

        if self.decode_q and self.use_question:
            # decode_q_final [batch_size, max_len, feature_size]
            decode_q_final = self._conv_max_pooling("q", self.sep_q_encodes, self.max_q_len)
            concat_arr.append(decode_q_final)

        if self.decode_r and self.use_response_u:
            # decode_r_final [batch_size, max_len, feature_size]
            decode_r_final = self._conv_max_pooling("r", self.sep_r_encodes, self.max_u_len)
            concat_arr.append(decode_r_final)

        # Combine all the pooled features
        # all arr in the concat arr are int the shape [batch_size, hidden_size]
        self.h_final = tf.concat(concat_arr, 1)
        # h_final is in the shape [batch_size, hidden_size1+hidden_size2, hidden_size3...]
        if self.use_dropout:
            self.h_final = tf.nn.dropout(self.h_final, self.dropout_keep_prob)

        self.scores = 0.0
        with tf.name_scope("decode_cnn"):
            final_state = self.h_final
            final_dim = self.feature_size
            final_state, final_dim = self._decode_extra_features(final_state, final_dim)
            self.scores += self._decode_func_multiclass(self.decode_func, self.num_classes, final_state, final_dim, "cnn")

    def _decode_with_seq_encodes(self):
        #[batch * max_window, num_classes]
        self.seq_scores = 0.0
        self.final_encodes = []
        self.final_encodes_dims = 0
        if 'seq_enc_final' in self.decode_inputs:
            # use the encodes
            # [batch_size * time, encode output_size]
            if 'bi' in self.snt_rnn_type:
                encodes_dim = self.state_size
            else:
                encodes_dim = self.hidden_size
            encodes = tf.reshape(self.sep_hp_encodes, [-1, encodes_dim])
            with tf.name_scope("decode_seq_encodes"):
                self.final_encodes_dims += encodes_dim
                self.final_encodes.append(encodes)
                if self.decode_combine == 'additive':
                    self.seq_scores += self._decode_func_multiclass(self.decode_func, self.num_classes, encodes, encodes_dim, "seq_encodes")

        # for decode_combine == concat, we concat all , then go through the fc layer.
        if self.decode_combine == 'concat':
            self.seq_scores += self._decode_func_multiclass(self.decode_func, self.num_classes, tf.concat(self.final_encodes, -1), self.final_encodes_dims, "concat_seq_all")

        # seq_scores reshaped into [batch_size, max_window, num_classes]
        if self.use_r_in_seq:
            seq_length = self.max_passage_window + 1
        else:
            seq_length = self.max_passage_window
        self.seq_scores = tf.reshape(self.seq_scores, [tf.shape(self.p_wsize)[0], seq_length, self.num_classes])

    def _decode_seq_tags(self):
        """
        decode with seq tags with crf 
        """
        if self.decode_group_size > 0:
            raise NotImplementedError('Unsupported cnn group for CRF')
        else:
            self._decode_with_seq_encodes()
            # self._decode_cnn_pooling_all()
            # self._decode_sim_WX_B()
        self._compute_seqtag_scores_and_loss()
        self._add_weight_decay_regularizer()

    def _decode_multiclass(self):
        """
        decode_multiclass
        """
        if self.decode_group_size > 0:
            self._decode_with_cnn_pooling()
        else:
            self._decode_with_final_state()
            # self._decode_cnn_pooling_all()
            # self._decode_sim_WX_B()
        self._compute_multiclass_pred_probs()
        self._compute_multiclass_loss()
        self._add_weight_decay_regularizer()

    def _compute_multiclass_pred_probs(self):
        self.pred_probs = tf.nn.softmax(self.scores)

    def _compute_seqtag_scores_and_loss(self):
        crf_params = tf.get_variable("crf_trans", [self.num_classes, self.num_classes], dtype=tf.float32)
        self.pred_seq_tags, self.crf_scores = tf.contrib.crf.crf_decode(self.seq_scores, crf_params, self.snt_size)
        self.reshaped_correct_seq_labels = tf.reshape(self.correct_seq_labels, [tf.shape(self.p_wsize)[0], -1])
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(self.seq_scores, self.reshaped_correct_seq_labels, self.snt_size, crf_params)
        self.loss = tf.reduce_mean(-log_likelihood)

    def _compute_multiclass_loss(self):
        """
        # Jie
        for various loss function
        """
        if self.loss_func == 'X_ENTROPY':
            self.loss = loss_zoo._compute_xentropy_with_logits(self.scores, self.correct_labels)
        elif self.loss_func == 'WEIGHTED_CE':
            # loss label weights.
            self.loss = loss_zoo._compute_weighted_xentropy_with_logits(self.scores, self.correct_labels, self.loss_weights)
        elif self.loss_func == 'WEIGHTED_FOCAL':
            # loss label weights.
            self.loss = loss_zoo._compute_weighted_focal_loss(self.scores,self.pred_probs, self.correct_labels, self.loss_weights, self.focal_loss_gama)
        else:
            raise NotImplementedError('The loss func {} is not implemented.'.format(self.loss_func))

    def _add_weight_decay_regularizer(self):
        """
        Apply weight_decay regularizer to loss
        """
        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))

        # allow gradient clipping before apply_gradients
        grads = self.optimizer.compute_gradients(self.loss)
        # use zip(*) to unzip a list into two.
        gradients, variables = zip(*grads)
        # @ISSUE clip gradidents by global_norm clipping.
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # the optimization of the training, zip back the clipped_gradients and vairbales.
        self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=self.global_step)

    def _train_epoch(self, epoch, train_batches, batch_size, data, dial_eval, min_loss, best_metrics, save_dir, save_prefix):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0

        # index start from 1, just let it not do evaluation at the start point.
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = self.feed_dict_to_placeholder(batch, 'train')
            if 'SEQTAG' in self.decode_goal:
                _, loss, pred_seq_tags = self.sess.run([self.train_op, self.loss, self.pred_seq_tags], feed_dict)
            else:
                #_, loss, pred_probs,scores = self.sess.run([self.train_op, self.loss, self.pred_probs, self.scores], feed_dict)
                _, loss, pred_probs = self.sess.run([self.train_op, self.loss, self.pred_probs], feed_dict)

            # self.logger.info('loss={}, pred_probs={}, scores={}'.format(loss, pred_probs, scores))
            # using number of answers
            # for binary, it is just the number of answers
            # for multiclass, it should be the number of psssages
            # However, passages * max_candidate_answers is also fine when computing the average.
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Epoch = {}, Average loss from batch {} to {} is {}'.format(
                    epoch,
                    bitx - log_every_n_batch + 1,
                    bitx,
                    n_batch_loss / log_every_n_batch)
                )
                n_batch_loss = 0

            # evaluate model and save occasionally
            # for steps_per_checkpoint, do evalution and save model.
            # global_step is number of batches, the tf has seen
            global_step = self.sess.run(self.global_step)
            if data is not None and global_step % self.steps_per_checkpoint == 0:
                dev_batches = data.gen_mini_batches('dev', batch_size, shuffle=False)
                self.logger.info('Evaluating the model in the epoch {}, after steps {}'.format(epoch, global_step))
                if self.use_ema:
                    self.save(save_dir, 'ema_temp')
                    with self.ema_test_graph.as_default():
                        if self.ema_test_model is None:
                            self.ema_test_model = DialogueModel(self.vocab, self.vocab_char, self.args)
                        self.ema_test_model.restore(save_dir, 'ema_temp', use_ema=True)
                    dev_loss, metrics = self.ema_test_model.evaluate(data.dev_context, dev_batches, dial_eval)
                else:
                    dev_loss, metrics = self.evaluate(data.dev_context, dev_batches, dial_eval)
                self.logger.info('Epoch = {}, Global_step = {}, Dev_eval_loss = {}'.format(epoch, global_step, dev_loss))
                if dev_loss < min_loss:
                    self.save(save_dir, save_prefix)
                    min_loss = dev_loss

                for k, v in metrics.items():
                    if k in best_metrics:
                        # v[0] is -1 or 1, to mean larger the better or smaller the better.
                        # v[1] is the actual value
                        if v[0] * v[1] > v[0] * best_metrics[k]:
                            self.logger.info("Epoch = {}, Global_step = {}, Update metric {} = {}".format(epoch, global_step, k, v[1]))
                            self.save(save_dir, save_prefix + "_{}".format(k))
                            best_metrics[k] = v[1]
                    else:
                        # unlike the loss, we just not set the best_metrics at first, then the first one always be savedn
                        self.logger.info("Epoch = {}, Global_step = {}, Update metric {} = {}".format(epoch, global_step, k, v[1]))
                        self.save(save_dir, save_prefix + "_{}".format(k))
                        best_metrics[k] = v[1]

        return 1.0 * total_loss / total_num, min_loss

    def feed_dict_to_placeholder(self, batch, phase_name):
        # feed_dict content:
        # p, the passage token ids, [#P, P_len]
        # q, the question token ids, pair with passage [#Q, Q_len]

        feed_dict = {
            self.p_wsize: batch['passage_window_size'],
            # [batch_size * max_passage_windown]
            self.p_u: batch['passage_utterance_token_ids'],
            self.p_u_speaker: batch['passage_utterance_speaker'],
            self.p_u_label: batch['passage_utterance_label'],
            self.p_u_length: batch['passage_utterance_length'],
            # [batch_size]
            self.q: batch['question_token_ids'],
            self.q_length: batch['question_length'],
            # [batch_size]
            self.r: batch['response_token_ids'],
            self.r_speaker: batch['response_speaker'],
            self.r_length: batch['response_length'],
        }

        if self.use_char_embedding:
            feed_dict[self.p_u_c] = batch['passage_utterance_char_ids']
            feed_dict[self.p_u_t_length] = batch['passage_utterance_token_length']
            feed_dict[self.qc] = batch['question_token_char_ids']
            feed_dict[self.q_t_length] = batch['question_token_length']

        if self.use_concat_p:
            # [batch_size]
            feed_dict[self.p] = batch['passage_token_ids']
            feed_dict[self.p_length] = batch['passage_length']
            if self.use_char_embedding:
                feed_dict[self.pc] = batch['passage_token_char_ids']
                feed_dict[self.p_t_length] = batch['passage_token_length']

        if self.elmo_utils:
            self.elmo_utils.prepare_elmo_cache_feed_dict_emb(feed_dict, batch)
            # self.elmo_utils.prepare_elmo_cache_feed_dict_sntids(feed_dict, batch)

        if phase_name == 'train' or phase_name == 'evaluate':
            feed_dict[self.correct_labels] = batch['correct_labels']
            feed_dict[self.correct_seq_labels] = batch['correct_seq_labels']

            if phase_name == 'train':
                feed_dict[self.dropout_keep_prob] = self.dropout_keep_prob_value
                feed_dict[self.dropout_keep_prob_emb] = self.dropout_keep_prob_emb_value
                feed_dict[self.dropout_keep_prob_mlp] = self.dropout_keep_prob_mlp_value
                feed_dict[self.is_training_phase] = True
            else:
                feed_dict[self.dropout_keep_prob] = 1.0
                feed_dict[self.dropout_keep_prob_emb] = 1.0
                feed_dict[self.dropout_keep_prob_mlp] = 1.0
                feed_dict[self.is_training_phase] = False
        else:
            feed_dict[self.dropout_keep_prob] = 1.0
            feed_dict[self.dropout_keep_prob_emb] = 1.0
            feed_dict[self.dropout_keep_prob_mlp] = 1.0
            feed_dict[self.is_training_phase] = False

        return feed_dict

    def train(self, data, epochs, batch_size, dial_eval, save_dir, save_prefix, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        min_loss = 9999
        best_metrics = {}
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            # for every epoch, just get the iterates for batches
            # Hence, every time calling the gen_mini_batches, it can reread all the files one by one.
            train_batches = data.gen_mini_batches('train', batch_size, shuffle=True)

            data1 = None
            if evaluate and data.dev_context is not None:
                data1 = data

            train_loss, min_loss = self._train_epoch(
                epoch,
                train_batches, batch_size, data1,
                dial_eval,
                min_loss,
                best_metrics,
                save_dir, save_prefix
            )

            self.logger.info(
                'Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if not evaluate:
                self.save(save_dir, save_prefix)

    def assign_seq_tags(self, batch, pred_seq_tags, predicted_answers):
        """
        according to the pred_seq_tags, set prob for predicted_answers
        """
        # for multiclass classifier, pred_prob is [batch_size, self.num_classes]
        # the score for every answer is in the correspoding index of the pred_prob
        for index in range(len(batch['raw_data'])):
            sample = batch['raw_data'][index]
            e_id = sample['example-id']
            predicted_sample = predicted_answers[e_id]
            predicted_sample['pred_seq_tags'] = pred_seq_tags[index].tolist()

    def assign_answer_prob(self, batch, pred_probs, predicted_answers):
        """
        according to the pred_probs, set prob for predicted_answers
        """
        # for multiclass classifier, pred_prob is [batch_size, self.num_classes]
        # the score for every answer is in the correspoding index of the pred_prob
        for index in range(len(batch['raw_data'])):
            sample = batch['raw_data'][index]
            e_id = sample['example-id']
            predicted_sample = predicted_answers[e_id]
            predicted_sample['pred_probs'] = []
            for i in range(self.num_classes):
                option = {}
                option['label_index'] = i
                option['label_name'] = self.classes[i]
                option['prob'] = float(pred_probs[index, i])
                predicted_sample['pred_probs'].append(option) 

    def transform_to_global_labelset(self, predicted_answers):
        """
        sub model usually only build for small subset of labels, here we transform them global model
        """
        global_labels = self.psyc_utils.All_labels
        for e_id, sample in predicted_answers.iteritems():
            local_pred_probs = sample['pred_probs']
            global_pred_probs = []
            for i in range(len(global_labels)):
                label = global_labels[i]
                option = {}
                option['label_index'] = i
                option['label_name'] = label
                option['prob'] = 0.0
                for old_option in local_pred_probs:
                    if old_option['label_name'] == label:
                        option['prob'] = old_option['prob']
                global_pred_probs.append(option)
            sample['pred_probs'] = global_pred_probs

    def evaluate(self, raw_context_dict, eval_batches, dial_eval):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
        """
        total_loss, total_num = 0, 0

        predicted_answers = {}

        # initialize the predicted_answers
        for e_id, sample in raw_context_dict.iteritems():
            predicted_answers[e_id] = sample
            sample.pop('pred_probs', None)
            sample.pop('pred_seq_tags', None)

        # for all flatten pairs (context, answer)in dev set.
        # first go through all flatten pairs, score all of the flatten pairs
        # then make top 10 selection based on the dataset and group up all the flatten pairs.

        for b_itx, batch in enumerate(eval_batches):
            feed_dict = self.feed_dict_to_placeholder(batch, "evaluate")
            if 'SEQTAG' in self.decode_goal:
                loss, pred_seq_tags = self.sess.run([self.loss, self.pred_seq_tags], feed_dict)
                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])
                # for binary: shape of pred_probs = [batch_size, 2]
                # for multiclass : shape of pred_probs = [batch_size, self.num_classes]
                self.assign_seq_tags(batch, pred_seq_tags, predicted_answers)
            else:
                if self.use_label_embedding or self.decode_func == 'BILINEAR':
                    loss, pred_probs, label_cm = self.sess.run([self.loss, self.pred_probs, self.label_cm], feed_dict)
                else:
                    loss, pred_probs = self.sess.run([self.loss, self.pred_probs], feed_dict)

                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])
                # for binary: shape of pred_probs = [batch_size, 2]
                # for multiclass : shape of pred_probs = [batch_size, self.num_classes]
                self.assign_answer_prob(batch, pred_probs, predicted_answers)

        # eval on different metrics
        metrics = dial_eval.eval_metrics(predicted_answers)
        if self.use_label_embedding or self.decode_func == 'BILINEAR':
            dial_eval.eval_label_confusion(label_cm)
        dial_eval.save_predicts(predicted_answers)
        ave_loss = 1.0 * total_loss / total_num
        return ave_loss, metrics

    def predict_without_save(self, raw_context_dict, test_batches, dial_eval):
        predicted_answers = {}

        # initialize the predicted_answers
        for e_id, sample in raw_context_dict.iteritems():
            predicted_answers[e_id] = sample
            sample.pop('pred_probs', None)
            sample.pop('pred_seq_tags', None)

        for b_itx, batch in enumerate(test_batches):
            feed_dict = self.feed_dict_to_placeholder(batch, 'train')
            if 'SEQTAG' in self.decode_goal:
                pred_seq_tags = self.sess.run([self.pred_seq_tags], feed_dict)
                self.assign_seq_tags(batch, pred_seq_tags, predicted_answers)
            else:
                if self.use_label_embedding or self.decode_func == 'BILINEAR':
                    pred_probs, label_cm = self.sess.run([self.pred_probs, self.label_cm], feed_dict)
                else:
                    pred_probs = self.sess.run([self.pred_probs], feed_dict)
                self.assign_answer_prob(batch, pred_probs, predicted_answers)

        return predicted_answers


    def predict(self, raw_context_dict, test_batches, dial_eval):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        predicted_answers = {}

        # initialize the predicted_answers
        for e_id, sample in raw_context_dict.iteritems():
            predicted_answers[e_id] = sample
            sample.pop('pred_probs', None)
            sample.pop('pred_seq_tags', None)

        for b_itx, batch in enumerate(test_batches):
            feed_dict = self.feed_dict_to_placeholder(batch, 'predict')
            if 'SEQTAG' in self.decode_goal:
                pred_seq_tags = self.sess.run([self.pred_seq_tags], feed_dict)
                self.assign_seq_tags(batch, pred_seq_tags, predicted_answers)
            else:
                if self.use_label_embedding or self.decode_func == 'BILINEAR':
                    pred_probs, label_cm = self.sess.run([self.pred_probs, self.label_cm], feed_dict)
                else:
                    pred_probs = self.sess.run([self.pred_probs], feed_dict)
                self.assign_answer_prob(batch, pred_probs, predicted_answers)

        dial_eval.save_predicts(predicted_answers)
        return predicted_answers


    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix, use_ema=None):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        if use_ema is None:
            use_ema = self.use_ema
        if use_ema:
            vars_ = {var.name.split(":")[0]: var for var in tf.global_variables()}
            for var in tf.trainable_variables():
                del vars_[var.name.split(":")[0]]
                vars_[self.ema.average_name(var)] = var
            temp_saver = tf.train.Saver(vars_)
            temp_saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        else:
            # path = os.path.join(model_dir, model_prefix+".meta")
            # self.logger.info("path is {}".format(path))
            # self.saver = tf.train.import_meta_graph(path)
            self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))

    def get_global_step(self):
        return self.global_step
