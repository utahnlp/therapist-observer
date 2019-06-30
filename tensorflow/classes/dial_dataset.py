# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 14:56:46 jcao>
# --------------------------------------------------------------------
# File Name          : dial_dataset.py
# Original Author    : jiessie.cao@gmail.com
# Description        : This model implements data process strategies
# --------------------------------------------------------------------

import ujson as json
import logging
import math
import numpy as np
import copy
import os
from collections import Counter
from collections import OrderedDict
import itertools
from elmo_utils import ELMo_Utils
from psyc_utils import PsycUtils
from feeddict_utils import padding

class DialogueDataset(object):

    """
    This module implements the API for loading dataset
    """
    def __init__(self, vocab, vocab_char, args, train_files=[], dev_files=[], test_files=[], elmo_utils=None):
        self.logger = logging.getLogger("dial")

        self.vocab = vocab
        self.vocab_char = vocab_char

        self.dataset_splits = {}
        self.dataset_splits['train'] = train_files
        self.dataset_splits['dev'] = dev_files
        self.dataset_splits['test'] = test_files

        # the class to class id lookup table
        self.pred_classes_mapping = {}

        #  the lookup table for all labels, which is the lookup tables for label embeddings and final prediction.
        self.all_label_mapping = {}
        self.decode_goal = args.decode_goal
        # exception labels will be involved when choose exception label strategy
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

        # only labels in pred_classes_mapping will be predicted, they are also used for speaker prediction.
        # P.S: When preding the labels, here id used for get label embedding is the same the in the final prediction class id.
        # For only preding P and T labels, pay attention to this, the ids may be not the same
        for i, x in enumerate(self.classes):
            self.pred_classes_mapping[x] = i

        # all label mapping will also add the padding label, and other context labels, which are not for predicting
        # P.S: When preding the labels, here id used for get label embedding is the same the in the final prediction class id.
        # For only preding P and T labels, pay attention to this, the ids may be not the same
        for j, x in enumerate(self.psyc_utils.All_labels):
            self.all_label_mapping[x] = j

        # padding label is not in the predicting label set
        # it is only used as features
        self.all_label_mapping[PsycUtils.PADDING_LABEL] = len(self.all_label_mapping)

        # these special ids are just at the begining of the vocab
        # their embedding are not in the pretrained embedding.
        self.pad_id = self.vocab.get_id(self.vocab.pad_token)
        self.sep_id = self.vocab.get_id(self.vocab.sep_token)
        self.sos_id = self.vocab.get_id(self.vocab.sos)
        self.eos_id = self.vocab.get_id(self.vocab.eos)
        if self.vocab_char:
            self.char_pad_id = self.vocab_char.get_id(self.vocab_char.pad_token)
            self.char_sep_id = self.vocab_char.get_id(self.vocab_char.sep_token)

        self.use_concat_p = args.use_concat_p
        # the concatanated  length of all context utterance
        self.max_p_len = args.max_p_len
        # the length of question, we use the last utterance as question currently
        self.max_q_len = args.max_q_len
        # max_length of a single utterance
        self.max_u_len = args.max_u_len

        self.speaker_set = {'T': 1, 'P': 2, PsycUtils.PADDING_SPEAKER: 0}

        # context window for context and question
        self.context_window = args.context_window
        self.use_r_in_seq = args.use_r_in_seq

        self.question_window = args.question_window

        # token_key_to_use for get the tokens
        self.token_key_to_use = args.token_key_to_use
        # self.token_key_to_use = 'tokenized_utterance'

        # may be discarded, we always use char_embedding
        self.use_char_embedding = args.use_char_embedding
        self.max_num_char_to_keep_forward = args.max_num_char_to_keep_forward
        self.max_num_char_to_keep_backward = args.max_num_char_to_keep_backward

        self.logger.info('Load and convert2id for training files {} lazily'.format(self.dataset_splits['train']))
        self.dev_context = OrderedDict()
        self.test_context = OrderedDict()
        if dev_files:
            for dev_file in self.dataset_splits['dev']:
                c = self._load_dataset(dev_file, 'dev')
                self.dev_context.update(c)

            self.logger.info('Dev all_dialogue_examples: {}'.format(len(self.dev_context)))

        if test_files:
            for test_file in self.dataset_splits['test']:
                c = self._load_dataset(test_file, 'test')
                self.test_context.update(c)

            self.logger.info('Test all_dialogue_examples: {}'.format(len(self.test_context)))

        self.elmo_utils = elmo_utils

    def _load_dataset(self, data_path, set_name):
        # for multiclassifier classifier, there is only one label
        # for multiple label classifier, there may exist multiple labels
        context_set = OrderedDict()
        with open(data_path, "r") as fin:
            for lidx, line in enumerate(fin):
                # current tokenized file is line by line no indent, so that we can load every example line-by-line
                sample = json.loads(line.strip())
                example_id = sample['example-id']
                # only feed P or T for special decode_goal
                if self.decode_goal == 'P_LABEL' and sample['options-for-correct-answers'][0]['speaker'] == 'T':
                    continue
                if self.decode_goal == 'T_LABEL' and sample['options-for-correct-answers'][0]['speaker'] == 'P':
                    continue
                # for train and dev set, we label every options according to the correct answers.
                # We didn't add answer label in the tokenized file, because different models will have different labels.
                if set_name in ['train', 'dev']:
                    # for multiclass labels, just add a single label for the whole dilogue, which is the index of the answer.
                    self.add_correct_label_multiclass(sample)
                    self.add_correct_seq_labels(sample)

                context_set[example_id] = sample

        self.logger.info('Converting text in ids, for split {}'.format(data_path))
        self.convert_to_ids(context_set)
        return context_set

    def _one_mini_batch(self, data, set_name, indices):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
        Returns:
            one batch of data
        """
        # here we assemly a full raw data sample by context and option into raw_data
        batch_data = {'raw_data': [data[i] for i in indices],
                      # [batch_size]
                      'passage_window_size': [],
                      # [batch_size * max_passage_windows_size, tokens]
                      'passage_utterance_token_ids': [],
                      # [batch_size * max_passage_window_size]
                      'passage_utterance_speaker': [],
                      # [batch_size * max_passage_window_size]
                      'passage_utterance_label': [],
                      # [batch_size * max_passage_window_size]
                      'passage_utterance_length': [],
                      # [batch_size, tokens]
                      'question_token_ids': [],
                      # [batch_size]
                      'question_length': [],
                      # [batch_size, tokens]
                      'response_token_ids': [],
                      # [batch_size]
                      'response_speaker': [],
                      # [batch_size]
                      'response_length': [],
                      # [batch_size]
                      'correct_labels': [],
                      # correct seq labels
                      # [batch_size * max_passage_window_size]
                      'correct_seq_labels': []
                      }

        if self.use_concat_p:
            batch_data['passage_token_ids'] = []
            # [batch_size]
            batch_data['passage_length'] = []

        if self.elmo_utils:
            batch_data['elmo_passage_char_ids'] = []
            batch_data['elmo_question_char_ids'] = []
            self.elmo_utils.init_elmo_batch_data_emb(batch_data)
            # self.elmo_utils.init_elmo_batch_data_sntids(batch_data)

        for idx, sample in enumerate(batch_data['raw_data']):
            example_id = sample['example-id']
            # prepapre regular feed_dict for question, passage, and answer
            self._prepare_question_feed_dict(sample, batch_data)
            self._prepare_passage_feed_dict(sample, batch_data)
            self._prepare_response_feed_dict(sample, batch_data)
            if self.elmo_utils:
                # prepare elmo feed_dict for question, passage, and answers
                self.elmo_utils._prepare_passage_elmo_feed_dict(sample, batch_data, self.context_window, self.token_key_to_use)
                self.elmo_utils._prepare_question_elmo_feed_dict(sample, batch_data, self.question_window, self.token_key_to_use)
                self.elmo_utils._prepare_response_elmo_feed_dict(sample, batch_data, self.token_key_to_use)

            # for labels, one label for per self.max_candidate_answers.
            if set_name != 'test':
                # Current only support multiclass
                # TODO for multiple labels
                # test may have the correct utterance when not for generating.
                batch_data['correct_labels'].append(sample['correct_labels'][0])

        # self._print_feed_dict_shape(batch_data)
        # padding for each utterance, passage, question, answers
        if self.elmo_utils:
            # for elmo, we force them into max_p_len, max_q_len, max_u_len
            self._static_padding(batch_data)
        else:
            self._dynamic_padding(batch_data)

        # self._print_feed_dict_shape(batch_data)

        # create character token ids
        # char tokens are [batch_size * self.max_candidate_answers * max_token_length]
        if self.use_char_embedding:
            if self.use_concat_p:
                tags = ['question', 'passage', 'response', 'passage_utterance']
            else:
                tags = ['question', 'response', 'passage_utterance']
            batch_data = self._add_char_tokens(batch_data, tags)

        return batch_data

    def _static_padding(self, batch_data):
        """
        This is used for static padding, which is useful when the deep contextual embedding is saved with a mask of the whole static length.
        """
        if self.use_concat_p:
            pad_p_len = self.max_p_len
            padding(batch_data, 'passage_token_ids', pad_p_len, self.pad_id)
            batch_data['passage_length'] = [pad_p_len if i>pad_p_len else i for i in batch_data['passage_length']]

        pad_q_len = self.max_q_len
        pad_pu_len = self.max_u_len
        pad_r_len = self.max_u_len

        # : ids is a list of word tokens of a most relevant paragraph in a doc
        padding(batch_data, 'question_token_ids', pad_q_len, self.pad_id)
        padding(batch_data, 'passage_utterance_token_ids', pad_pu_len, self.pad_id)
        padding(batch_data, 'response_token_ids', pad_r_len, self.pad_id)

        # adjust length according to conform to ids
        batch_data['question_length'] = [pad_q_len if i>pad_q_len else i for i in batch_data['question_length']]
        batch_data['response_length'] = [pad_r_len if i>pad_r_len else i for i in batch_data['response_length']]
        batch_data['passage_utterance_length'] = [pad_pu_len if i>pad_pu_len else i for i in batch_data['passage_utterance_length']]

    def _print_feed_dict_shape(self, batch_data):
        for k, v in batch_data.iteritems():
            self.logger.info("shape for {} is {}".format(k, np.shape(v)))
            #self.logger.info("shape for {} is {}".format(k, v))

    def _prepare_question_feed_dict(self, sample, batch_data):
        """
        given a sample, prepare the related feed dict for question
        """
        # prepare for questions
        # question may also can be represented with hierachy bi-lstm
        # use question window to decide how many sentense are selected as question content
        question_token_ids = [self.sos_id]
        pruned_question_utterances = sample['messages-so-far'][-self.question_window:]
        for i in range(self.question_window):
            if i >= len(pruned_question_utterances):
                pass
            else:
                utterance = pruned_question_utterances[i]
                # concatenated version of question
                # append question utterance tokens
                question_token_ids.extend(utterance['utterance_token_ids'])

        question_token_ids.extend([self.eos_id])
        # add question
        # [batch_size, question_length]
        batch_data['question_token_ids'].append(question_token_ids)
        # real length of question, which will be used for dynamic padding
        # [batch_size]
        batch_data['question_length'].append(len(question_token_ids))

    def _prepare_response_feed_dict(self, sample, batch_data):
        '''
        Given a sample, and batch_data, feed_dict, we extract the passage info rom the sample, and assemble the feed dict for tranining model.
        '''
        # prepare for passages
        # use context_window to decide how many sentences are selected as context content
        if 'options-for-correct-answers' in sample:
            utterance = sample['options-for-correct-answers'][0]
            response_token_ids = [self.sos_id]
            response_token_ids.extend(utterance['utterance_token_ids'])
            response_token_ids.extend([self.eos_id])
            batch_data['response_token_ids'].append(response_token_ids)
            batch_data['response_speaker'].append(self.speaker_set[utterance['speaker']])
            batch_data['response_length'].append(len(response_token_ids))

    def _prepare_passage_feed_dict(self, sample, batch_data):
        '''
        Given a sample, and batch_data, feed_dict, we extract the passage info rom the sample, and assemble the feed dict for tranining model.
        '''
        # prepare for passages
        # use context_window to decide how many sentences are selected as context content

        pruned_context_utterances = sample['messages-so-far'][-self.context_window:]
        passage_utterance_token_ids = []
        passage_utterance_speaker = []
        passage_utterance_label = []
        passage_utterance_length = []
        passage_token_ids = [self.sos_id]
        for i in range(self.context_window):
            if i >= len(pruned_context_utterances):
                # not enough to read context_window, add empty padding
                current_utterance_token_ids = [self.sos_id, self.eos_id]
                passage_utterance_token_ids.append(current_utterance_token_ids)
                passage_utterance_speaker.append(self.speaker_set[PsycUtils.PADDING_SPEAKER])
                passage_utterance_label.append(self.all_label_mapping[PsycUtils.PADDING_LABEL])
                passage_utterance_length.append(len(current_utterance_token_ids))
            else:
                utterance = pruned_context_utterances[i]
                # for splitted version
                current_utterance_token_ids = [self.sos_id]
                current_utterance_token_ids.extend(utterance['utterance_token_ids'])
                current_utterance_token_ids.extend([self.eos_id])
                passage_utterance_token_ids.append(current_utterance_token_ids)
                passage_utterance_speaker.append(self.speaker_set[utterance['speaker']])
                # add label
                # for response, it is not allowed to use utterance label later.
                most_common_label = self.psyc_utils.get_most_common_label(utterance)
                passage_utterance_label.append(self.all_label_mapping[most_common_label])
                passage_utterance_length.append(len(current_utterance_token_ids))
                # for concated version
                # append utterance tokens
                passage_token_ids.extend(utterance['utterance_token_ids'])
        passage_token_ids.extend([self.eos_id])
        # if we add padding during generating the dataset, then allpadding will become the real data.
        actual_passage_window_size = len(pruned_context_utterances)
        batch_data['passage_window_size'].append(actual_passage_window_size)
        if self.use_concat_p:
            # for now, we concact all the context utternance into a single passag
            # [batch_size, passage_length]
            batch_data['passage_token_ids'].append(passage_token_ids)
            # [batch_size]
            batch_data['passage_length'].append(len(passage_token_ids))
        # add passage
        # for now, we concact all the context utternance into a single passag
        # each utterance, flatten them into batch_size ()
        # [batch_size * passage_window, utterance_length]
        batch_data['passage_utterance_token_ids'].extend(passage_utterance_token_ids)
        # [batch_size * passage_window]
        batch_data['passage_utterance_speaker'].extend(passage_utterance_speaker)
        # [batch_size * passage_window]
        batch_data['passage_utterance_label'].extend(passage_utterance_label)
        # [batch_size * passage_window]
        batch_data['correct_seq_labels'].extend(sample['correct_seq_labels'])
        # [batch_size * passage_window]
        batch_data['passage_utterance_length'].extend(passage_utterance_length)

    # Add character tokens into the paded word token ids
    # Treat each word tokenlen(e)feed a 2-d tensor char token id like in word token id tensor
    # we also use character embedding for answering
    def _add_char_tokens(self, batch_data, tags=['passage', 'passage_utterance','question']):
        # TODO: add tags for passage_utterance, question_utterance
        """
        add char tokens for corresponding tags.
        Especially, answer is a little different tensor.
        """
        for tag in tags:
            word_key = '{}_token_ids'.format(tag)
            char_key = '{}_token_char_ids'.format(tag)
            char_len_key = '{}_token_length'.format(tag)
            batch_data[char_key] = []
            batch_data[char_len_key] = []
            char_tokens = []  # this should be 2-dimensional tensor like in word token id
            char_token_len = []  # this should be 1-dimensional tensor like in word token id
            for wids in batch_data[word_key]:
                for wid in wids:
                    if wid == self.pad_id:
                        # size of character to use forward + backward
                        char_tokens.append([self.char_pad_id] * (self.max_num_char_to_keep_forward + self.max_num_char_to_keep_backward))
                        # Would seq length = 0 be problematic in tensorflow LSTM?
                        # since the pad word is masked anyway, it is okay to fill in the number here
                        char_token_len.append(0)
                    else:
                        token = self.vocab.recover_from_ids([wid])
                        # convert word into char ids list
                        forward_chars = list(token)[:self.max_num_char_to_keep_forward]
                        backward_chars = list(token)[-self.max_num_char_to_keep_backward:]
                        forward_char_ids = self.vocab_char.convert_to_ids(forward_chars)
                        backward_char_ids = self.vocab_char.convert_to_ids(backward_chars)
                        # pad char_ids
                        padded_char_ids = (forward_char_ids + [self.char_pad_id] * (self.max_num_char_to_keep_forward - len(forward_char_ids) + self.max_num_char_to_keep_backward - len(backward_char_ids)) + backward_char_ids)[:(self.max_num_char_to_keep_forward + self.max_num_char_to_keep_backward)]
                        char_tokens.append(padded_char_ids)
                        char_token_len.append(np.sum(np.not_equal(padded_char_ids, self.char_pad_id)))
            # create record
            # char_ley and char_len_key is [batch_size*self.max_candidate_answers*max_token_lenth, max_char_length]
            batch_data[char_key] = char_tokens
            batch_data[char_len_key] = char_token_len
        return batch_data

    def _dynamic_padding(self, batch_data):
        """
        Dynamically pads the batch_data with pad_id
        """
        # just use min(self.max_p_len, max(lengh for whole batch)
        if self.use_concat_p:
            pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
            padding(batch_data, 'passage_token_ids', pad_p_len, self.pad_id)
            batch_data['passage_length'] = [pad_p_len if i>pad_p_len else i for i in batch_data['passage_length']]

        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        padding(batch_data, 'question_token_ids', pad_q_len, self.pad_id)

        # since we may concat u and pu all together, we chose the max of them.
        pad_pu_len = min(self.max_u_len, max(batch_data['passage_utterance_length']))
        pad_r_len = min(self.max_u_len, max(batch_data['response_length']))
        pad_rpu_common_len = max(pad_pu_len, pad_r_len)
        padding(batch_data, 'passage_utterance_token_ids', pad_rpu_common_len, self.pad_id)
        padding(batch_data, 'response_token_ids', pad_rpu_common_len, self.pad_id)

        # adjust length according to conform to ids
        batch_data['question_length'] = [pad_q_len if i>pad_q_len else i for i in batch_data['question_length']]
        batch_data['response_length'] = [pad_rpu_common_len if i>pad_r_len else i for i in batch_data['response_length']]
        batch_data['passage_utterance_length'] = [pad_rpu_common_len if i>pad_pu_len else i for i in batch_data['passage_utterance_length']]

    # for multiclass classifier, the whole example will only have a single label
    def add_correct_label_multiclass(self, sample):
        """
        according to the correct_answers in the sample, add a binary label for all the answers.
        """
        correct_answers = sample['options-for-correct-answers']
        if 'SPEAKER' in self.decode_goal:
            sample['correct_labels'] = map(lambda u : self.pred_classes_mapping[u['speaker']], correct_answers)
        elif any(map(lambda x: x in self.decode_goal, ['ALL_LABEL', 'P_LABEL', 'T_LABEL'])):
            correct_labels = map(lambda u: self.pred_classes_mapping[u['agg_label']] , correct_answers)
            sample['correct_labels'] = correct_labels
        else:
            raise NotImplementedError("{} is not supported".format(self.decode_goal))

    def add_correct_seq_labels(self, sample):
        label_seq = []
        pruned_context_utterances = sample['messages-so-far'][-self.context_window:]
        passage_utterance_label = []
        for i in range(self.context_window):
            if i >= len(pruned_context_utterances):
                # not enough to read context_window, add empty padding
                passage_utterance_label.append(self.all_label_mapping[PsycUtils.PADDING_LABEL])
            else:
                utterance = pruned_context_utterances[i]
                # add label
                most_common_label = self.psyc_utils.get_most_common_label(utterance)
                passage_utterance_label.append(self.all_label_mapping[most_common_label])
        if self.use_r_in_seq:
            if 'options-for-correct-answers' in sample:
                uc = sample['options-for-correct-answers'][0]
                correct_response_label = self.pred_classes_mapping[uc['agg_label']]
                label_seq.extend(passage_utterance_label)
                label_seq.append(correct_response_label)
            else:
                raise RuntimeError("No options-for-correct-answers")
        else:
            label_seq.extend(passage_utterance_label)

        sample['correct_seq_labels'] = label_seq

    def gen_dataset_splits(self, set_name, shuffle=False):
        """
        just generate data split for this set_name
        every time, it will only load one split of the dataset
        """
        if set_name in self.dataset_splits:
            dataset_splits_files = self.dataset_splits[set_name]
            if shuffle:
                dataset_splits_files = np.random.permutation(self.dataset_splits[set_name])
            else:
                dataset_splits_files = self.dataset_splits[set_name]
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

        # just return the cached dev and test set
        if set_name == 'dev':
            yield self.dev_context
        elif set_name == 'test':
            yield self.test_context
        else:
            for split_file in dataset_splits_files:
                yield self._load_dataset(split_file, set_name)

    def word_iter(self, set_name):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        for context in self.gen_dataset_splits(set_name):
            # for context
            for _, sample in context.iteritems():
                for context_utterance in sample['messages-so-far']:
                    for token in context_utterance[self.token_key_to_use]:
                        yield token
                for correct_utterance in sample['options-for-correct-answers']:
                    for token in correct_utterance[self.token_key_to_use]:
                        yield token

    def convert_to_ids(self, context):
        """
        convert all the utterance in the original dataset to ids
        args:
        vocab: the vocabulary on this dataset
        """
        for _, sample in context.iteritems():
            for context_utterance in sample['messages-so-far']:
                context_utterance['utterance_token_ids'] = self.vocab.convert_to_ids(context_utterance[self.token_key_to_use])
            for correct_utterance in sample['options-for-correct-answers']:
                correct_utterance['utterance_token_ids'] = self.vocab.convert_to_ids(correct_utterance[self.token_key_to_use])

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        """
        One example in multiclass classifier, is one context with mutiple candidate answers, the label for this example is just the index of the label. Which is added in _load_dataset.
        This function will return a generator for every example

        """
        for context in self.gen_dataset_splits(set_name):
            context_arr = [c for _, c in context.iteritems()]
            data_size = len(context_arr)
            indices = np.arange(data_size)
            if shuffle:
                # do inner shuffle
                np.random.shuffle(indices)
                # context_arr is a array of the raw data with all candidate in its canidate answers.
                # selected_indices are all the indices in the global context array
            for batch_start in np.arange(0, data_size, batch_size):
                selected_indices = indices[batch_start: batch_start + batch_size]
                yield self._one_mini_batch(context_arr, set_name, selected_indices)

if __name__ == '__main__':
    dataset = DialogueDataset
