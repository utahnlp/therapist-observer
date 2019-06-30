# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 15:02:38 jcao>
# --------------------------------------------------------------------
# File Name          : elmo_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : This module is about elmo.
# --------------------------------------------------------------------

import os
import time
import ujson as json
import numpy as np
import h5py
import snt_dict
from tqdm import tqdm
from bilm import Batcher, TokenBatcher, BidirectionalLanguageModel, weight_layers
import tensorflow as tf
import logging
from io_utils import get_num_lines
from feeddict_utils import padding

class ELMo_Utils(object):
    """
    Impements Elmo functions used by downstream task
    Each tokenized sentence is a list of str, with a batch of sentences a list of tokenized sentences (List[List[str]]).

The Batcher packs these into a shape (n_sentences, max_sentence_length + 2, 50) numpy array of character ids, padding on the right with 0 ids for sentences less then the maximum length. The first and last tokens for each sentence are special begin and end of sentence ids added by the Batcher.

The input character id placeholder can be dimensioned (None, None, 50), with both the batch dimension (axis=0) and time dimension (axis=1) determined for each batch, up the the maximum batch size specified in the BidirectionalLanguageModel constructor.

After running inference with the batch, the return biLM embeddings are a numpy array with shape (n_sentences, 3, max_sentence_length, 1024), after removing the special begin/end tokens.
    """

    START_TOKEN = '<S>'
    END_TOKEN = '</S>'
    UNK_TOKEN = '<UNK>'
    PAD_SNT ='<S></S>'
    PAD_SNT_ID = 0

    def __init__(self, elmo_vocab_file, elmo_weight_file, elmo_option_file, use_character_elmo, use_concat_p, question_window, utterance_cache_file='', passage_cache_file='', question_cache_file=''):
        self.logger = logging.getLogger("dial")
        self.utterance_cache = None
        self.passage_cache = None
        self.question_cache = None
        self.need_q_cache = (question_window > 1)
        self.need_p_cache = use_concat_p
        if os.path.exists(elmo_weight_file) and os.path.exists(elmo_option_file) and os.path.exists(elmo_vocab_file):
            # the vocab file exported from the corpus
            self.elmo_vocab_file = elmo_vocab_file
            # elmo weight file
            self.elmo_weight_file = elmo_weight_file
            # elmo option file
            self.elmo_option_file = elmo_option_file
            self.utterance_cache_file = utterance_cache_file
            self.passage_cache_file = passage_cache_file
            self.question_cache_file = question_cache_file
            self.use_character_elmo = use_character_elmo
            with open(self.elmo_option_file, 'r') as fin:
                options = json.load(fin)
            self.output_layers = options['lstm']['n_layers'] + 1
            self.output_dim =  2 * options['lstm']['projection_dim']
            self.logger.info("output_layers :{}, output_dim :{}".format(self.output_layers, self.output_dim))
            # by default, the bilm use the character_elmo
            if self.use_character_elmo:
                # max_num_char for characters for a token.
                self.elmo_max_num_char = options['char_cnn']['max_characters_per_token']
                # line 207 https://github.com/allenai/bilm-tf/blob/ebf52c6ec1012a3672247c2d14ff7bcad7fb812b/bilm/data.py
                # the mask for char id is 0
                self.PAD_TOKEN_CHAR_IDS = np.zeros((self.elmo_max_num_char), dtype=np.int32).tolist()
                # use subword character first, which shows extra improvements beside the contextual information.
                self.elmo_char_batcher = Batcher(self.elmo_vocab_file, self.elmo_max_num_char)
                # language mode with use_character_inputs = True
                self.elmo_bilm = BidirectionalLanguageModel(self.elmo_option_file, self.elmo_weight_file)
            else:
                # use token batcher
                self.elmo_token_batcher = TokenBatcher(self.elmo_vocab_file)
                # use elmo_bilm with use_character_inputs = False
                self.elmo_bilm = BidirectionalLanguageModel(self.elmo_option_file, self.elmo_weight_file)

            self.chk_load_utterance_cache()
            self.chk_load_passage_cache()
            self.chk_load_question_cache()
        else:
            self.logger.warn("elmo_weight_file = {}, elmo_option_file={}, elmo_vocab_file={}".format(elmo_weight_file, elmo_option_file, elmo_vocab_file))

    def chk_load_utterance_cache(self):
        if self.utterance_cache_file and os.path.exists(self.utterance_cache_file):
            self.utterance_cache = h5py.File(self.utterance_cache_file, 'r')
            #self.utterance_cache_in_mem = {}
            #self.utterance_cache_in_mem['lm_embeddings'] = self.load_h5(self.utterance_cache['lm_embeddings'])
            #self.utterance_cache_in_mem['lengths'] = self.load_h5_lengths(self.utterance_cache['lengths'])
            #self.utterance_cache_in_mem['mask'] = self.load_h5(self.utterance_cache['mask'])
            self.logger.info("Utterance cache loaded from {}, size = {}".format(self.utterance_cache_file, len(self.utterance_cache['lm_embeddings'].keys())))
        else:
            self.utterance_cache = None

    def load_h5(self, h5group):
        x = []
        for index in range(len(h5group.keys())):
            # https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
            x.append(h5group['{}'.format(index)][...].tolist())
        return x

    def load_h5_lengths(self, h5group):
        x = []
        for index in range(len(h5group.keys())):
            x.extend(h5group['{}'.format(index)][...].tolist())
        return x


    def chk_load_passage_cache(self):
        if self.need_p_cache:
            if self.passage_cache_file and os.path.exists(self.passage_cache_file):
                self.passage_cache = h5py.File(self.passage_cache_file, 'r')
                self.logger.info("Passage cache loaded from {}".format(self.passage_cache_file))
            else:
                self.passage_cache = None
                self.logger.info("Passage cache needed from {}, it will build soon.".format(self.passage_cache_file))
        else:
            self.passage_cache = None
            self.logger.info("Passage cache not needed")

    def chk_load_question_cache(self):
        if self.need_q_cache:
            if self.question_cache_file and os.path.exists(self.question_cache_file):
                self.question_cache = h5py.File(self.question_cache_file, 'r')
                self.logger.info("Question cache loaded from {}".format(self.question_cache_file))
            else:
                self.question_cache = None
                self.logger.info("Question cache needed from {}, it will build soon.".format(self.question_cache_file))
        else:
            self.question_cache = None
            self.logger.info("Question cache not needed")

    def need_build_passage_cache(self):
        return self.need_p_cache and self.passage_cache_file != '' and self.passage_cache == None

    def need_build_question_cache(self):
        return self.need_q_cache and self.question_cache_file != '' and self.question_cache == None

    def cleanup(self):
        if self.utterance_cache:
            self.utterance_cache.close()
        if self.passage_cache:
            self.passage_cache.close()
        if self.question_cache:
            self.question_cache.close()
        self.logger.info("Clean up elmo cahce")

    def get_elmo_char_ids(self, sentences):
        '''
        Given a nested list of tokens(with start and end token), return the character ids
        Arguments:
            sentences: List[List[str]]

        Return: [sentence_num, token_num, max_char_num]
        '''
        return self.elmo_char_batcher.batch_sentences(sentences).tolist()


    def get_elmo_token_ids(self, sentences):
        '''
        Given a nested list of tokens(without start and end token), return the token ids

        Arguments:
           sentemces : List[List[str]]

        Return : [sentence_num, token_num, max_char_num]
        '''
        return self.elmo_token_batcher.batch_sentences(sentences).tolist()

    def get_elmo_emb_op(self, input_ids_place_holder):
        '''
        Given the input ids place holder, reutrn a ops for computing the language model
        {
         'lm_embeddings': embedding_op, (None, 3, None, 1024)
         'lengths': sequence_lengths_op, (None, )
         'mask': op to compute mask (None, None)
        }
        '''
        return self.elmo_bilm(input_ids_place_holder)

    def weight_layers(self, name, bilm_ops, l2_coef=None, use_top_only=False, do_layer_norm=False):
        '''
        Weight the layers of a biLM with trainable scalar weights to compute ELMo representations.
        See more details on https://github.com/allenai/bilm-tf/blob/81a4b54937f4dfb93308f709c1cf34dbb37c553e/bilm/elmo.py
        {
           'weighted_op': op to compute weighted average for output,
           'regularization_op': op to compute regularization term
        }
        '''
        return weight_layers(name, bilm_ops, l2_coef, use_top_only, do_layer_norm)

    @staticmethod
    def prepare_elmo_vocab_file(vocab, elmo_vocab_file):
        sorted_word = sorted(vocab.token_cnt, key = vocab.token_cnt.get, reverse = True)
        with open(elmo_vocab_file, 'w') as f:
            f.write('{}\n'.format(ELMo_Utils.START_TOKEN))
            f.write('{}\n'.format(ELMo_Utils.END_TOKEN))
            f.write('{}\n'.format(ELMo_Utils.UNK_TOKEN))
            for item in sorted_word:
                f.write('%s\n' % item)


    def build_elmo_char_cache(self, snt_dict_file, max_snt_length, output_cache_file):
        """
        Go through all the snts in the dataset, save into the cache
        """
        self.logger.info('Prepare ELMo character embeddings for {} with ELMo_Utils ...'.format(snt_dict_file))
        ids_placeholder = tf.placeholder('int32', shape=(None, max_snt_length, self.elmo_max_num_char))
        ops = self.elmo_bilm(ids_placeholder)
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            with open(snt_dict_file, 'r') as fin, h5py.File(output_cache_file, 'w') as fout:
                lm_embeddings_h5 = fout.create_group('lm_embeddings')
                lengths_h5 = fout.create_group('lengths')
                mask_h5 = fout.create_group('mask')
                batch_snts = []
                start_snt_id_in_batch = 0
                SNT_BATCH_SIZE = 10
                for line in tqdm(fin, total=get_num_lines(snt_dict_file)):
                    sentence = line.strip().split()
                    batch_snts.append(sentence)
                    length = len(batch_snts)
                    if length >= SNT_BATCH_SIZE:
                        start_snt_id_in_batch += self.consume_batch_snts(sess, ids_placeholder, ops, batch_snts,max_snt_length, start_snt_id_in_batch, lm_embeddings_h5, lengths_h5, mask_h5)
                        batch_snts = []
                if len(batch_snts) > 0:
                    start_snt_id_in_batch += self.consume_batch_snts(sess, ids_placeholder, ops, batch_snts,max_snt_length, start_snt_id_in_batch, lm_embeddings_h5, lengths_h5, mask_h5)
                    batch_snts = []
                self.logger.info("Finished ELMo embeddings for {} senencesm in {}".format(start_snt_id_in_batch, output_cache_file))

    def consume_batch_snts(
            self, sess, ids_placeholder, ops,
            batch_snts, max_snt_length, start_snt_id_in_batch,
            lm_embeddings_h5, lengths_h5, mask_h5
    ):
        char_ids = self.get_elmo_char_ids(batch_snts)
        char_ids = [(ids + [self.PAD_TOKEN_CHAR_IDS] * (max_snt_length - len(ids)))[:max_snt_length] for ids in char_ids]
        elmo_ops = sess.run(
            ops, feed_dict={ids_placeholder: char_ids}
        )
        batch_size = len(batch_snts)
        for i in range(batch_size):
            sentence_id = start_snt_id_in_batch + i
            # self.logger.info("create lm for snt {}".format(sentence_id))
            lm_embeddings_h5.create_dataset(
                '{}'.format(sentence_id),
                elmo_ops['lm_embeddings'].shape[1:], dtype='float32',
                data=elmo_ops['lm_embeddings'][i, :, :, :], compression="gzip"
            )
            lengths_h5.create_dataset(
                '{}'.format(sentence_id),
                (1,), dtype='int32',
                data=elmo_ops['lengths'][i]
            )
            mask_h5.create_dataset(
                '{}'.format(sentence_id),
                elmo_ops['mask'].shape[1:], dtype='int32',
                data=elmo_ops['mask'][i], compression="gzip"
            )
        return batch_size

    # TODO for token level embedding.
    def build_elmo_token_cache(self, snt_dict_file, max_snt_length, output_cache_file):
        pass

    def build_elmo_cache(self, snt_dict_file, max_snt_length, output_cache_file):
        if self.use_character_elmo:
            self.build_elmo_char_cache(snt_dict_file, max_snt_length, output_cache_file)
        else:
            self.build_elmo_token_cache(snt_dict_file, max_snt_length, output_cache_file)

        self.logger.info('Finished ELMo embeddings for utterance cache with ELMo_Utils')

    def build_elmo_cache_for_samples(self, dataset, max_p_len, max_q_len):
        if (not self.need_p_cache) and (not self.need_q_cache):
            self.logger.info('No need for ELMo embeddings for concated passage and question with ELMo_Utils')
        else:
            # build graph for getting forward elmo embedding.
            self.logger.info('Build ELMo embeddings for p = {}, q = {}'.format(self.need_p_cache, self.need_q_cache))
            self.build_pq_elmo_graph()
            if self.need_p_cache:
                p_out = h5py.File(self.passage_cache_file, 'w')
                p_lm_embeddings_h5 = p_out.create_group('lm_embeddings')
                p_lengths_h5 = p_out.create_group('lengths')
                p_mask_h5 = p_out.create_group('mask')

            if self.need_q_cache:
                q_out = h5py.File(self.question_cache_file, 'w')
                q_lm_embeddings_h5 = q_out.create_group('lm_embeddings')
                q_lengths_h5 = q_out.create_group('lengths')
                q_mask_h5 = q_out.create_group('mask')

            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                for set_name in ['train', 'dev', 'test']:
                    for batch_data in tqdm(dataset.gen_mini_batches(set_name, 20, shuffle=False)):
                        samples = batch_data['raw_data']
                        # batch_data is filled with elmo feed_dict
                        self.run_pq_ops(sess, batch_data, max_p_len, max_q_len)
                        for i in range(len(samples)):
                            e_id = '{}'.format(samples[i]['example-id'])
                            try:
                                if self.need_p_cache:
                                    p_lm_embeddings_h5.create_dataset(
                                        e_id,
                                        p_ops['lm_embeddings'].shape[1:], dtype='float32',
                                        data=p_ops['lm_embeddings'][i, :, :, :], compression="gzip"
                                    )
                                    p_lengths_h5.create_dataset(
                                        e_id,
                                        (1,), dtype='int32',
                                        data=p_ops['lengths'][i]
                                    )
                                    p_mask_h5.create_dataset(
                                        e_id,
                                        p_ops['mask'].shape[1:], dtype='int32',
                                        data=p_ops['mask'][i, :], compression="gzip"
                                    )
                                if self.need_q_cache:
                                    q_lm_embeddings_h5.create_dataset(
                                        e_id,
                                        q_ops['lm_embeddings'].shape[1:], dtype='float32',
                                        data=q_ops['lm_embeddings'][i, :, :, :],compression="gzip"
                                    )
                                    q_lengths_h5.create_dataset(
                                        e_id,
                                        (1,), dtype='int32',
                                        data=q_ops['lengths'][i],
                                    )
                                    q_mask_h5.create_dataset(
                                        e_id,
                                        q_ops['mask'].shape[1:], dtype='int32',
                                        data=q_ops['mask'][i, :], compression="gzip"
                                    )
                            except:
                                continue

        self.logger.info('Finished ELMo embeddings for concated passage and question with ELMo_Utils')

    def run_pq_ops(self, sess, batch_data, max_p_len, max_q_len):
        self._static_pq_padding(batch_data, max_p_len, max_q_len)

        if self.need_p_cache and self.need_q_cache:
            self.p_ops, self.q_ops = sess.run([self.p_emb_elmo_op, self.q_emb_elmo_op],
                                    feed_dict=
                                 {
                                     self.elmo_p : batch_data['elmo_passage_char_ids'],
                                     self.elmo_q : batch_data['elmo_question_char_ids']
                                 })
        elif self.need_p_cache:
            self.p_ops = sess.run([self.p_emb_elmo_op],
                                    feed_dict=
                                 {
                                     self.elmo_p : batch_data['elmo_passage_char_ids']
                                 })
        else:
            self.q_ops = sess.run([self.q_emb_elmo_op],
                                    feed_dict=
                                 {
                                     self.elmo_q : batch_data['elmo_question_char_ids'],
                                 })

    def build_pq_elmo_graph(self):
        """
        Given the batch_data, this will seperately run tensorflow get the elmo embedding for each batch, which will be cached into file
        Especially , for sample level cache, please make sure that the first dimension for any tensor is batch_size
        """
        start_t = time.time()
        self.logger.info("Start building elmo graph for concatenated p and q ...")
        self.add_elmo_placeholders()
        with tf.device('/device:GPU:0'):
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                # get all elmo op with language mode
                # lm_embeddings : [batch_size, layers, max_length, hidden_dims * 2]
                # lengths : [batch_size]
                # mask : [batch_size, length]
                if self.need_p_cache:
                    self.p_emb_elmo_op = self.elmo_bilm(self.elmo_p)

                if self.need_q_cache:
                # [batch_size, context_window, layers, max_u_length, hidden_dims * 2]
                    self.q_emb_elmo_op = self.elmo_bilm(self.elmo_q)

    def add_elmo_placeholders(self):
        """
        elmo for business, logic corresponding the specific application
        """
        # for ELMo with character embedding
        # elmo passage character ids for each token in each concatenated passage
        # [batch_size, passage_length, char_length]

        if self.need_p_cache:
            self.elmo_p = tf.placeholder(tf.int32, [None, None, self.elmo_max_num_char], 'elmo_p')
        # elmo character ids for whole concatenated qustion
        # [batch_size, question_length, char_length]
        self.elmo_q = tf.placeholder(tf.int32, [None, None, self.elmo_max_num_char], 'elmo_q')

    def _static_pq_padding(self, batch_data, max_p_len, max_q_len):
        """
        This is used for static padding, which is useful when the deep contextual embedding is saved with a mask of the whole static length.
        """
        # also padding elmo matrix
        # in elmo, the character ids after batch_sentences contains the start and end token, length for charids +2 while the final embedding not contains those special token.
        # For further compatibility, we still leave elmo length as different length.
        pad_q_len_elmo = 2 + max_q_len
        padding(batch_data, 'elmo_question_char_ids', pad_q_len_elmo, self.PAD_TOKEN_CHAR_IDS)

        if self.need_p_cache:
            pad_p_len_elmo = 2 + max_p_len
            padding(batch_data, 'elmo_passage_char_ids', pad_p_len_elmo, self.PAD_TOKEN_CHAR_IDS)

    def _prepare_passage_elmo_feed_dict(self, sample, batch_data, context_window, token_key_to_use):
        """
        add elmo feed_dict for passage
        """
        e_id_str = '{}'.format(sample['example-id'])
        passage_utterance_tokens_elmo = []
        passage_utterance_length_elmo = []
        passage_tokens_elmo = [ELMo_Utils.START_TOKEN]
        passage_snt_ids = []
        pruned_context_utterances_elmo = sample['messages-so-far'][-context_window:]
        for i in range(context_window):
            if i >= len(pruned_context_utterances_elmo):
                current_utterance_tokens_elmo = [ELMo_Utils.START_TOKEN, ELMo_Utils.END_TOKEN]
                passage_snt_ids.append(ELMo_Utils.PAD_SNT_ID)
                passage_utterance_tokens_elmo.append(current_utterance_tokens_elmo)
                passage_utterance_length_elmo.append(len(current_utterance_tokens_elmo))
            else:
                utterance = pruned_context_utterances_elmo[i]
                if 'snt_id' in utterance:
                    passage_snt_ids.append(utterance['snt_id'])
                # split version of passages
                current_utterance_tokens_elmo = [ELMo_Utils.START_TOKEN]
                current_utterance_tokens_elmo.extend(utterance[token_key_to_use])
                current_utterance_tokens_elmo.extend([ELMo_Utils.END_TOKEN])
                passage_utterance_tokens_elmo.append(current_utterance_tokens_elmo)
                passage_utterance_length_elmo.append(len(current_utterance_tokens_elmo))
                # concatenated version of passages
                # append passages utterance tokens
                passage_tokens_elmo.extend(utterance[token_key_to_use])

        passage_tokens_elmo.extend([ELMo_Utils.END_TOKEN])
        if self.need_build_passage_cache():
            # add into batch_data, no other batch data will data
            # [batch_size, passage_length, max_char_num]
            batch_data['elmo_passage_char_ids'].append(self.get_elmo_char_ids([passage_tokens_elmo])[0])
        else:
            #TODO add passage and question elmo retrieve here.
            if self.need_p_cache:
                self.assemble_elmo_batch_data('p', batch_data, e_id_str, self.passage_cache)
            for snt_id in passage_snt_ids:
                # self.assemble_elmo_with_snt_ids('pu', batch_data, snt_id)
                # self.assemble_elmo_batch_data_with_mem('pu', batch_data, snt_id, self.utterance_cache_in_mem)
                self.assemble_elmo_batch_data('pu', batch_data, snt_id, self.utterance_cache)

    def _prepare_question_elmo_feed_dict(self, sample, batch_data, question_window, token_key_to_use):
        """
        add question elmo feed_dict according the same style for adding regular question feed_dict
        """
        e_id_str = '{}'.format(sample['example-id'])
        # for each utterance in question
        question_utterance_tokens_elmo = []
        # for the concatenated question
        # for question utterance length
        question_utterance_length_elmo = []
        question_snt_ids = []
        # add start token, which is also in the vocabulary
        # in non-elmo, embedding, we wil add self.vocab.sos and self.vocab.eos in to the sentence,whic will be encoded by the downstream lstm. However, sos and eos are in capital case in the elmo. In fact, we must use Upper case here to get a emebdding from elmo abou it.
        question_tokens_elmo = [ELMo_Utils.START_TOKEN]
        pruned_question_utterance_elmo = sample['messages-so-far'][-question_window:]
        for i in range(question_window):
            if i >= len(pruned_question_utterance_elmo):
                current_utterance_tokens_elmo = [ELMo_Utils.START_TOKEN, ELMo_Utils.END_TOKEN]
                question_snt_ids.append(ELMo_Utils.PAD_SNT_ID)
                question_utterance_tokens_elmo.append(current_utterance_tokens_elmo)
                question_utterance_length_elmo.append(len(current_utterance_tokens_elmo))
            else:
                utterance = pruned_question_utterance_elmo[i]
                # split version of question
                if 'snt_id' in utterance:
                    question_snt_ids.append(utterance['snt_id'])
                current_utterance_tokens_elmo = [ELMo_Utils.START_TOKEN]
                current_utterance_tokens_elmo.extend(utterance[token_key_to_use])
                current_utterance_tokens_elmo.extend([ELMo_Utils.END_TOKEN])
                # add each utterance token_ids into a parental list
                question_utterance_tokens_elmo.append(current_utterance_tokens_elmo)
                question_utterance_length_elmo.append(len(current_utterance_tokens_elmo))
                # concatenated version of question
                # append question utterance tokens
                question_tokens_elmo.extend(utterance[token_key_to_use])

        question_tokens_elmo.extend([ELMo_Utils.END_TOKEN])
        if question_window == 0:
            # if note use question, here it will make mistake,
            # bug here. make question at least = 1
            pass
        else:
            # add elmo question tokenids into batch_data
            if self.need_build_question_cache():
                # add into batch_data
                # [batch_size, question_length, max_char_num]
                batch_data['elmo_question_char_ids'].append(self.get_elmo_char_ids([question_tokens_elmo])[0])
            else:
                # if question_window = 1, then juse use utterance cache
                if question_window == 1:
                    # self.assemble_elmo_with_snt_ids('q', batch_data, question_snt_ids[0])
                    # self.assemble_elmo_batch_data_with_mem('q', batch_data, question_snt_ids[0], self.utterance_cache_in_mem)
                    self.assemble_elmo_batch_data('q', batch_data, question_snt_ids[0], self.utterance_cache)
                else:
                    self.assemble_elmo_batch_data('q', batch_data, e_id_str, self.question_cache)

    def _prepare_response_elmo_feed_dict(self, sample, batch_data, token_key_to_use):
        """
        add question elmo feed_dict according the same style for adding regular question feed_dict
        """
        if 'options-for-correct-answers':
            e_id_str = '{}'.format(sample['example-id'])
            utterance = sample['options-for-correct-answers'][0]
            # split version of question
            current_utterance_tokens_elmo = [ELMo_Utils.START_TOKEN]
            current_utterance_tokens_elmo.extend(utterance[token_key_to_use])
            current_utterance_tokens_elmo.extend([ELMo_Utils.END_TOKEN])
            if 'snt_id' in utterance:
                snt_id = utterance['snt_id']
                self.assemble_elmo_batch_data('r', batch_data, snt_id, self.utterance_cache)

    def init_elmo_batch_data_sntids(self, batch_data):
        if self.need_p_cache:
            # use elmo cache to retrieve batch_data
            batch_data['elmo_p_lm_embeddings'] = []
            batch_data['elmo_p_lengths'] = []
            batch_data['elmo_p_mask'] = []
        batch_data['elmo_pu_snt_ids'] = []
        batch_data['elmo_q_snt_ids'] = []
        batch_data['elmo_r_snt_ids'] = []

    def init_elmo_batch_data_emb(self, batch_data):
        if self.need_p_cache:
            # use elmo cache to retrieve batch_data
            batch_data['elmo_p_lm_embeddings'] = []
            batch_data['elmo_p_lengths'] = []
            batch_data['elmo_p_mask'] = []

        # for passage_utterance
        batch_data['elmo_pu_lm_embeddings'] = []
        batch_data['elmo_pu_lengths'] = []
        batch_data['elmo_pu_mask'] = []
        # for question
        batch_data['elmo_q_lm_embeddings'] = []
        batch_data['elmo_q_lengths'] = []
        batch_data['elmo_q_mask'] = []
        # for res
        batch_data['elmo_r_lm_embeddings'] = []
        batch_data['elmo_r_lengths'] = []
        batch_data['elmo_r_mask'] = []

    def add_elmo_placeholder_with_cache_sntids(self):
        """
        add placeholders for elmo ops, which will be used in the weight_layers
        """
        if self.need_p_cache:
            self.elmo_p_lm_embeddings = tf.placeholder(tf.float32, [None, self.output_layers, None, self.output_dim], name='elmp_p_lm_embeddings')
            self.elmo_p_lengths = tf.placeholder(tf.int32, [None], name='elmo_p_lengths')
            self.elmo_p_mask = tf.placeholder(tf.int32, [None, None], name='elmo_p_mask')

        self.elmo_pu_snt_ids = tf.placeholder(tf.int32, [None], name='elmo_pu_snt_ids')
        self.elmo_q_snt_ids = tf.placeholder(tf.int32, [None], name='elmo_q_snt_ids')
        self.elmo_r_snt_ids = tf.placeholder(tf.int32, [None], name='elmo_r_snt_ids')

    def add_elmo_placeholder_with_cache_emb(self):
        """
        add placeholders for elmo ops, which will be used in the weight_layers
        """
        if self.need_p_cache:
            self.elmo_p_lm_embeddings = tf.placeholder(tf.float32, [None, self.output_layers, None, self.output_dim], name='elmp_p_lm_embeddings')
            self.elmo_p_lengths = tf.placeholder(tf.int32, [None], name='elmo_p_lengths')
            self.elmo_p_mask = tf.placeholder(tf.int32, [None, None], name='elmo_p_mask')

        self.elmo_pu_lm_embeddings = tf.placeholder(tf.float32, [None, self.output_layers, None, self.output_dim], name='elmo_pu_lm_embeddings')
        self.elmo_pu_lengths = tf.placeholder(tf.int32, [None], name='elmo_pu_lengths')
        self.elmo_pu_mask = tf.placeholder(tf.int32, [None, None], name='elmo_pu_mask')
        self.elmo_q_lm_embeddings = tf.placeholder(tf.float32, [None, self.output_layers, None, self.output_dim], name='elmo_q_lm_embeddings')
        self.elmo_q_lengths = tf.placeholder(tf.int32, [None], name='elmo_q_lengths')
        self.elmo_q_mask = tf.placeholder(tf.int32, [None, None], name='elmo_q_mask')
        self.elmo_r_lm_embeddings = tf.placeholder(tf.float32, [None, self.output_layers, None, self.output_dim], name='elmo_r_lm_embeddings')
        self.elmo_r_lengths = tf.placeholder(tf.int32, [None], name='elmo_r_lengths')
        self.elmo_r_mask = tf.placeholder(tf.int32, [None, None], name='elmo_r_mask')

    def prepare_elmo_cache_feed_dict_sntids(self, feed_dict, batch):
        """
        consitently feed the batch_data, we prepared in the prepare_passage_elmo, question_elmo, answer_elmo
        """
        if self.need_p_cache:
            # for elmo_p
            feed_dict[self.elmo_p_lm_embeddings] = batch['elmo_p_lm_embeddings']
            feed_dict[self.elmo_p_lengths] = batch['elmo_p_lengths']
            feed_dict[self.elmo_p_mask] = batch['elmo_p_mask']

        # for elmo_q
        feed_dict[self.elmo_q_snt_ids] = batch['elmo_q_snt_ids']
        # for elmo_pu
        feed_dict[self.elmo_pu_snt_ids] = batch['elmo_pu_snt_ids']
        # for elmo_r
        feed_dict[self.elmo_r_snt_ids] = batch['elmo_r_snt_ids']

    def prepare_elmo_cache_feed_dict_emb(self, feed_dict, batch):
        """
        consitently feed the batch_data, we prepared in the prepare_passage_elmo, question_elmo, answer_elmo
        """
        if self.need_p_cache:
            # for elmo_p
            feed_dict[self.elmo_p_lm_embeddings] = batch['elmo_p_lm_embeddings']
            feed_dict[self.elmo_p_lengths] = batch['elmo_p_lengths']
            feed_dict[self.elmo_p_mask] = batch['elmo_p_mask']

        # for elmo_q
        feed_dict[self.elmo_q_lm_embeddings] = batch['elmo_q_lm_embeddings']
        feed_dict[self.elmo_q_lengths] = batch['elmo_q_lengths']
        feed_dict[self.elmo_q_mask] = batch['elmo_q_mask']

        # for elmo_pu
        feed_dict[self.elmo_pu_lm_embeddings] = batch['elmo_pu_lm_embeddings']
        feed_dict[self.elmo_pu_lengths] = batch['elmo_pu_lengths']
        feed_dict[self.elmo_pu_mask] = batch['elmo_pu_mask']

        # for elmo_r
        feed_dict[self.elmo_r_lm_embeddings] = batch['elmo_r_lm_embeddings']
        feed_dict[self.elmo_r_lengths] = batch['elmo_r_lengths']
        feed_dict[self.elmo_r_mask] = batch['elmo_r_mask']

    def elmo_embedding_layer_emb(self, elmo_emb_output):
        """
        elmo embedding layers, which will return embedding for p,q,a,pu,qu
        after projections, dim is elmo_emb_output
        if elmo_emb_output == self.output_dim, then no projection will be done
        """
        self.logger.info('build elmo embedding layer')
        if self.need_p_cache:
            p_emb_elmo_op = {
                'lm_embeddings': self.elmo_p_lm_embeddings,
                'lengths': self.elmo_p_lengths,
                'mask': self.elmo_p_mask
            }

        q_emb_elmo_op = {
            'lm_embeddings': self.elmo_q_lm_embeddings,
            'lengths': self.elmo_q_lengths,
            'mask': self.elmo_q_mask
        }

        pu_emb_elmo_op = {
            'lm_embeddings': self.elmo_pu_lm_embeddings,
            'lengths': self.elmo_pu_lengths,
            'mask': self.elmo_pu_mask
        }

        r_emb_elmo_op = {
            'lm_embeddings': self.elmo_r_lm_embeddings,
            'lengths': self.elmo_r_lengths,
            'mask': self.elmo_r_mask
        }

        with tf.device('/device:GPU:1'):
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                if self.need_p_cache:
                    self.p_elmo_emb = self.weight_layers('input', p_emb_elmo_op, l2_coef=0.0)['weighted_op']
                self.q_elmo_emb = self.weight_layers('input', q_emb_elmo_op, l2_coef=0.0)['weighted_op']
                self.pu_elmo_emb = self.weight_layers('input', pu_emb_elmo_op, l2_coef=0.0)['weighted_op']
                self.r_elmo_emb = self.weight_layers('input', r_emb_elmo_op, l2_coef=0.0)['weighted_op']
                # do project from elmo embedding into 128 embedding to contact with word embedding.
                if elmo_emb_output == self.output_dim:
                    self.logger.info("Elmo_emb_output={} is just equal to the output_dim={}, no need to project with fully connected layers for passage and questions".format(elmo_emb_output, self.output_dim))
                else:
                    self.logger.info("Elmo_emb_output={}, output_dim={}, project with fully connected layers for question and passage".format(elmo_emb_output, self.output_dim))
                    if self.need_p_cache:
                        self.p_elmo_emb = tf.contrib.layers.fully_connected(inputs=self.p_elmo_emb, num_outputs=elmo_emb_output, activation_fn=tf.nn.softmax)

                    self.q_elmo_emb = tf.contrib.layers.fully_connected(inputs=self.q_elmo_emb, num_outputs=elmo_emb_output, activation_fn=tf.nn.softmax)
                    self.pu_elmo_emb = tf.contrib.layers.fully_connected(inputs=self.pu_elmo_emb, num_outputs=elmo_emb_output, activation_fn=tf.nn.softmax)
                    self.r_elmo_emb = tf.contrib.layers.fully_connected(inputs=self.r_elmo_emb, num_outputs=elmo_emb_output, activation_fn=tf.nn.softmax)

    def elmo_embedding_layer_sntids(self, elmo_emb_output):
        """
        elmo embedding layers, which will return embedding for p,q,a,pu,qu
        after projections, dim is elmo_emb_output
        if elmo_emb_output == self.output_dim, then no projection will be done
        """
        with tf.device('/cpu:0'), tf.variable_scope('elmo_embedding'):
            self.elmo_lm_embeddings_lookup = tf.get_variable(
                'lm_embeddings_lookup',
                shape=np.shape(self.utterance_cache_in_mem['lm_embeddings']),
                initializer=tf.constant_initializer(self.utterance_cache_in_mem['lm_embeddings']),
                trainable=False
            )

            self.elmo_lengths_lookup = tf.get_variable(
                'lengths_lookup',
                shape=(np.shape(self.utterance_cache_in_mem['lengths'])),
                initializer=tf.constant_initializer(self.utterance_cache_in_mem['lengths']),
                trainable=False
            )

            self.elmo_mask_lookup = tf.get_variable(
                'mask_lookup',
                shape=np.shape(self.utterance_cache_in_mem['mask']),
                initializer=tf.constant_initializer(self.utterance_cache_in_mem['mask']),
                trainable=False
            )

        if self.need_p_cache:
            p_emb_elmo_op = {
                'lm_embeddings': self.elmo_p_embeddings,
                'lengths': self.elmo_p_lengths,
                'mask': self.elmo_p_mask
            }

        q_emb_elmo_op = {
            'lm_embeddings': tf.nn.embedding_lookup(self.elmo_lm_embeddings_lookup, self.elmo_q_snt_ids),
            'lengths': tf.nn.embedding_lookup(self.elmo_lengths_lookup, self.elmo_q_snt_ids),
            'mask': tf.nn.embedding_lookup(self.elmo_mask_lookup, self.elmo_q_snt_ids)
        }

        pu_emb_elmo_op = {
            'lm_embeddings': tf.nn.embedding_lookup(self.elmo_lm_embeddings_lookup, self.elmo_pu_snt_ids),
            'lengths': tf.nn.embedding_lookup(self.elmo_lengths_lookup, self.elmo_pu_snt_ids),
            'mask': tf.nn.embedding_lookup(self.elmo_mask_lookup, self.elmo_pu_snt_ids)
        }

        r_emb_elmo_op = {
            'lm_embeddings': tf.nn.embedding_lookup(self.elmo_lm_embeddings_lookup, self.elmo_r_snt_ids),
            'lengths': tf.nn.embedding_lookup(self.elmo_lengths_lookup, self.elmo_r_snt_ids),
            'mask': tf.nn.embedding_lookup(self.elmo_mask_lookup, self.elmo_r_snt_ids)
        }

        with tf.device('/device:GPU:1'):
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                if self.need_p_cache:
                    self.p_elmo_emb = self.weight_layers('input', p_emb_elmo_op, l2_coef=0.0)['weighted_op']
                self.q_elmo_emb = self.weight_layers('input', q_emb_elmo_op, l2_coef=0.0)['weighted_op']
                self.pu_elmo_emb = self.weight_layers('input', pu_emb_elmo_op, l2_coef=0.0)['weighted_op']
                self.r_elmo_emb = self.weight_layers('input', r_emb_elmo_op, l2_coef=0.0)['weighted_op']
                # do project from elmo embedding into 128 embedding to contact with word embedding.
                if elmo_emb_output == self.output_dim:
                    self.logger.info("Elmo_emb_output={} is just equal to the output_dim={}, no need to project with fully connected layers for question and passage".format(elmo_emb_output, self.output_dim))
                else:
                    self.logger.info("Elmo_emb_output={}, output_dim={}, project with fully connected layers for question and passage".format(elmo_emb_output, self.output_dim))
                    if self.need_p_cache:
                        self.p_elmo_emb = tf.contrib.layers.fully_connected(inputs=self.p_elmo_emb, num_outputs=elmo_emb_output, activation_fn=tf.nn.softmax)

                    self.q_elmo_emb = tf.contrib.layers.fully_connected(inputs=self.q_elmo_emb, num_outputs=elmo_emb_output, activation_fn=tf.nn.softmax)
                    self.pu_elmo_emb = tf.contrib.layers.fully_connected(inputs=self.pu_elmo_emb, num_outputs=elmo_emb_output, activation_fn=tf.nn.softmax)
                    self.r_elmo_emb = tf.contrib.layers.fully_connected(inputs=self.r_elmo_emb, num_outputs=elmo_emb_output, activation_fn=tf.nn.softmax)

    def assemble_elmo_batch_data(self, name, batch_data, id_key, cache):
        lm_embeddings = cache['lm_embeddings']['{}'.format(id_key)][...]
        length = cache['lengths']['{}'.format(id_key)][0]
        mask = cache['mask']['{}'.format(id_key)][...]
        batch_data['elmo_{}_lm_embeddings'.format(name)].append(lm_embeddings)
        batch_data['elmo_{}_lengths'.format(name)].append(length)
        batch_data['elmo_{}_mask'.format(name)].append(mask)

    def assemble_elmo_batch_data_with_mem(self, name, batch_data, id_key, cache_in_mem):
        """
        id_key is int here, for the snt_id
        """
        lm_embeddings = cache_in_mem['lm_embeddings'][id_key]
        length = cache_in_mem['lengths'][id_key]
        mask = cache_in_mem['mask'][id_key]
        batch_data['elmo_{}_lm_embeddings'.format(name)].append(lm_embeddings)
        batch_data['elmo_{}_lengths'.format(name)].append(length)
        batch_data['elmo_{}_mask'.format(name)].append(mask)

    def assemble_elmo_with_snt_ids(self, name, batch_data, id_key):
        """
        id_key is int here, for the snt_id
        """
        batch_data['elmo_{}_snt_ids'.format(name)].append(id_key)
