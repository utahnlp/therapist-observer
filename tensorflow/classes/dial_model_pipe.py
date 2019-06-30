# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 15:02:03 jcao>
# --------------------------------------------------------------------
# File Name          : dial_model_pipe.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A pipeline model mimic whole dialogue from each part
# --------------------------------------------------------------------

import os
import time
import logging
import re
import json as json
import pickle
import numpy as np
import copy
import tensorflow as tf
from tqdm import tqdm
import math_utils
import tf_utils
from psyc_utils import PsycUtils
from elmo_utils import ELMo_Utils
import config_reader
import loss_zoo
from dial_model import DialogueModel
from layers.basic_rnn import rnn
# from layers.memory_network import memory_network_rnn_multi_mem
# from layers.gated_memory_network import gated_memory_network_rnn_multi_mem
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.match_layer import GatedAttentionLayer
from layers.match_layer import CollaborativeGatedAttentionLayer
from layers.match_layer import GatedMatchLSTMLayer
from layers.match_layer import CollaborativeGatedMatchLSTMLayer
from layers.match_layer import multihead_attention, dot_product_attention, scaled_dot_product_attention

class DialogueModelPipe(object):
    """
    Implements the main dialogue model according to the preivous rc_model.
    """
    def __init__(self, vocab, vocab_char, args, elmo_utils=None):
        # logging
        self.logger = logging.getLogger("dial")
        self.psyc_utils = PsycUtils(args.cluster_strategy)
        self.context_window = args.context_window
        self.decode_goal = args.decode_goal

        #  the lookup table for all labels, which is the lookup tables for label embeddings and final prediction.
        self.all_label_mapping = {}

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

        for j, x in enumerate(self.psyc_utils.All_labels):
            self.all_label_mapping[x] = j

        # padding label is not in the predicting label set
        # it is only used as features
        self.all_label_mapping[PsycUtils.PADDING_LABEL] = len(self.all_label_mapping)

        self.num_classes = len(self.classes)
        self.pred_classes_mapping = {}

        for i, x in enumerate(self.classes):
            self.pred_classes_mapping[x] = i

        # load differet models by reading differnt model config files
        self.P_model = self._restore_model_with_config_file(args.p_model_config)
        self.T_model = self._restore_model_with_config_file(args.t_model_config)

    """
    read config from P_MODEL_CONFIG, T_MODEL_CONFIG
    create two different dialogue models instance and vocabularies.
    """
    def _restore_model_with_config_file(self, config_file):
        sub_args = config_reader.get_args(config_file)
        self.logger.info('Load data_set and vocab... for sub model. {}'.format(config_file))
        with open(os.path.join(sub_args.vocab_dir, 'vocab.data'), 'rb') as fin:
            vocab = pickle.load(fin)
        assert len(sub_args.dev_files) > 0, 'No dev files are provided.'
        if sub_args.use_char_embedding:
            with open(os.path.join(sub_args.vocab_dir, 'vocab_char.data'), 'rb') as fin:
                vocab_char = pickle.load(fin)
        else:
            vocab_char = None

        if len(sub_args.elmo_positions):
            self.logger.info('Initializing ELMo Utils from submodels {}'.format(config_file))
            elmo_inst = ELMo_Utils(sub_args.elmo_vocab_file, sub_args.elmo_weight_file, sub_args.elmo_option_file, sub_args.use_character_elmo, sub_args.use_concat_p, sub_args.question_window, sub_args.elmo_u_cache_file, sub_args.elmo_p_cache_file, sub_args.elmo_q_cache_file)
        else:
            elmo_inst = None

        # create a differnt vocabularies (vocab_char , vocab )that can be used in a new model instance
        model = DialogueModel(vocab, vocab_char, sub_args, elmo_utils=elmo_inst)
        model_to_restore = sub_args.model_prefix_to_restore if sub_args.model_prefix_to_restore else sub_args.algo
        model.restore(model_dir=sub_args.model_dir, model_prefix=model_to_restore)
        return model

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
                option['prob'] = float(pred_probs[index, i])
                predicted_sample['pred_probs'].append(option)

    def assign_answer_prob_with_samples(self, samples, predicted_answers, predicted_uid_labels):
        """
        according to the pred_probs, set prob for predicted_answers
        """
        # for multiclass classifier, pred_prob is [batch_size, self.num_classes]
        # the score for every answer is in the correspoding index of the pred_prob
        for e_id, sample in samples.iteritems():
            u_id = sample['options-for-correct-answers'][0]['uid']
            predicted_sample = predicted_answers[e_id]
            predicted_sample['pred_probs'] = sample['pred_probs']
            sorted_probs = sorted(predicted_sample['pred_probs'], reverse=True, key=lambda x: x['prob'])
            top1_pred = sorted_probs[0]
            # store the predicted uid_lables by u_id
            predicted_uid_labels[u_id] = top1_pred['label_name']


    def evaluate(self, raw_context_dict, eval_batches, dial_eval):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
        """
        total_loss, total_num = 0, 0

        # <eid, sample>
        predicted_answers = {}

        # <utteranceid , labels>
        # adding init PADDING_U_ID and PADDING LABEL as its label
        predicted_uid_labels = {PsycUtils.PADDING_U_ID : PsycUtils.PADDING_LABEL}

        # initialize the predicted_answers
        for e_id, sample in raw_context_dict.iteritems():
            predicted_answers[e_id] = sample
            sample.pop('pred_probs', None)
            sample.pop('pred_seq_tags', None)

        # for all flatten pairs (context, answer)in dev set.
        # first go through all flatten pairs, score all of the flatten pairs
        # then make top 10 selection based on the dataset and group up all the flatten pairs.

        # TODO: make sure it is in order, and batch_size = 1
        for b_itx, batch in enumerate(eval_batches):
            # eval on different metrics
            assert len(batch['correct_labels']) == 1
            if batch['response_speaker'][0] == 1:
                # therapist
                model = self.T_model
            elif batch['response_speaker'][0] == 2:
                model = self.P_model
            else:
                raise RuntimeError("response speaker is neright T or P")
            # need to modify all the labels in the batch data with previous prediction.
            e_id = batch['raw_data'][0]['example-id']
            old_sample = predicted_answers[e_id]
            #self.logger.info("b_itx = {}, old sample is {}".format(b_itx, json.dumps(old_sample, indent=4, encoding="utf-8")))
            updated_sample = self.update_context_labels(old_sample, batch, predicted_uid_labels)
            # construct a dict with only one example
            updated_1example_dict = {e_id: updated_sample}
            # self.logger.info("b_itx = {}, updated sample is {}".format(b_itx, json.dumps(updated_1example_dict, indent=4, encoding="utf-8")))
            # update the batch, if the batch is updated, then it will be fine.
            # do prediction with correponding model
            new_predicted_samples = model.predict_without_save(updated_1example_dict, [batch], dial_eval)
            # new_predicted_sampes has only 1 example. but with pred_probs and labels from the P_model or T_model, we need to change them into our model with All labels.
            #  we only use the label from the new_predicted_sample
            model.transform_to_global_labelset(new_predicted_samples)
            self.assign_answer_prob_with_samples(new_predicted_samples, predicted_answers, predicted_uid_labels)

        metrics = dial_eval.eval_metrics(predicted_answers)
        dial_eval.save_predicts(predicted_answers)
        ave_loss = 1.0 * total_loss / total_num
        return ave_loss, metrics

    def update_context_labels(self, old_sample, batch, predicted_uid_labels):
        for u in old_sample['messages-so-far']:
            # find the uid of u first
            uid = u['uid']
            oracel_label = u['agg_label']
            if uid in predicted_uid_labels:
                u['agg_label'] = predicted_uid_labels[uid]
            else:
                raise NotImplementedError("uid={} has not been predicted".format(uid))
            u['oracel_label'] = oracel_label

        pruned_context_utterances = old_sample['messages-so-far'][-self.context_window:]
        passage_utterance_label = []
        for i in range(self.context_window):
            if i >= len(pruned_context_utterances):
                # not enough to read context_window, add empty padding
                passage_utterance_label.append(self.all_label_mapping[PsycUtils.PADDING_LABEL])
            else:
                utterance = pruned_context_utterances[i]
                most_common_label = self.psyc_utils.get_most_common_label(utterance)
                passage_utterance_label.append(self.all_label_mapping[most_common_label])

        # [batch_size * passage_window]
        batch['passage_utterance_label'] = passage_utterance_label
        return old_sample

    # Wenxi
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
