# -*- coding:utf8 -*-
# Time-stamp: <2019-06-04 14:48:47 jcao>
# --------------------------------------------------------------------
# File Name          : dial_run.py
# Original Author    : jiessie.cao@gmail.com
# Description        : This module prepares and runs the whole system.
# --------------------------------------------------------------------

import sys
import os
import pickle
import argparse
import logging
import ujson as json
import tqdm
from classes.dial_dataset import DialogueDataset
from classes.vocab import Vocab
from classes.dial_model import DialogueModel
from classes.dial_model_pipe import DialogueModelPipe
from classes.dial_eval import DialogueEval
from classes.elmo_utils import ELMo_Utils
import classes.config_reader as config_reader
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """

    logger = logging.getLogger("dial")

    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)

    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    # word embeding vocab
    vocab = Vocab(lower=True)

    # character embedding vocab
    vocab_char = Vocab(lower=True)

    dial_data = DialogueDataset(
        vocab,
        vocab_char,
        args,
        args.train_files,
        args.dev_files,
        args.test_files
    )

    # the special tokens should be normalized in the specific dialogue.
    # word_iter only interate the common words.
    # for special normalization, we should have a special word_iter
    # "tokenized_utterance" in the preprocessing may not be directlly used for training.

    # for train, dev, test, field words for every one
    #for set_name in ['train', 'dev', 'test']:
    for set_name in ['train', 'dev']:
        for word in dial_data.word_iter(set_name):
            vocab.add(word)
            if args.use_char_embedding:
                for c in set(list(word)):
                    vocab_char.add(c)

    unfiltered_vocab_size = vocab.size()
    # not filter when sanity check make it overfitting .
    vocab.filter_tokens_by_cnt(min_cnt=2)

    if args.elmo_positions :
        logger.info('ELMo will be used in {}'.format(args.elmo_positions))
        if args.elmo_snt_dict_file and os.path.exists(args.elmo_snt_dict_file):
            logger.info('Sorted and write vocab.token_cnt into {}'.format(args.elmo_vocab_file))
            ELMo_Utils.prepare_elmo_vocab_file(vocab, args.elmo_vocab_file)
            elmo_inst = ELMo_Utils(args.elmo_vocab_file, args.elmo_weight_file, args.elmo_option_file, args.use_character_elmo, args.use_concat_p, args.question_window, args.elmo_u_cache_file, args.elmo_p_cache_file, args.elmo_q_cache_file)
            dial_data.elmo_utils = elmo_inst
            # build cache for all snts, include pu, qu, a
            if not elmo_inst.utterance_cache:
                elmo_inst.build_elmo_cache(args.elmo_snt_dict_file, args.max_u_len + 2, args.elmo_u_cache_file)
                elmo_inst.chk_load_utterance_cache()
            # build cache for concated p and q
            if not (elmo_inst.passage_cache and elmo_inst.question_cache):
                elmo_inst.build_elmo_cache_for_samples(dial_data, args.max_p_len, args.max_q_len)
                elmo_inst.chk_load_passage_cache()
                elmo_inst.chk_load_question_cache()
        else:
            logger.error("elmo_snt_dict_file = {} required for preparing elmo cache !".format(args.elmo_snt_dict_file))

    filtered_vocab_size = vocab.size()
    filtered_num = unfiltered_vocab_size - filtered_vocab_size
    logger.info('After frequency filter {} tokens, the final vocab size is {}'.format(filtered_num, filtered_vocab_size))
    logger.info('Assigning embeddings...')
    # Need to ensure that the pretrained embedding can cover all training vocab
    if args.word_emb_file is None or not args.word_emb_file:
        logger.info('Use Randomized embeddings...')
        vocab.randomly_init_embeddings(args.word_embed_size)
    else:
        logger.info('Use pre-trained embeddings...')
        vocab.load_pretrained_embeddings(args.word_emb_file)

    logger.info('After frequency filter {} tokens, the vocab size is {}, finally {} not in pre-trained embeddings'.format(filtered_num, filtered_vocab_size, filtered_vocab_size - vocab.size()))
    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    if args.use_char_embedding:
        if args.char_emb_file is None or not args.char_emb_file:
            logger.info('Use Randomized embeddings...')
            vocab_char.randomly_init_embeddings(args.char_embed_size)
        else:
            logger.info('Use pre-trained char embeddings...')
            vocab_char.load_pretrained_embeddings(args.char_emb_file)
        with open(os.path.join(args.vocab_dir, 'vocab_char.data'), 'wb') as fout:
            pickle.dump(vocab_char, fout)

    # prepare for the whole dataset, and write them into a h5 file.
    # then every mini batch, it will not to assemble the batch_data again,
    # padding will take it slightly different for those mask and padding for each sentences.

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("dial")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    if args.use_char_embedding:
        with open(os.path.join(args.vocab_dir, 'vocab_char.data'), 'rb') as fin:
            vocab_char = pickle.load(fin)
    else:
        vocab_char = None

    if len(args.elmo_positions):
        logger.info('Initializing ELMo Utils ...')
        elmo_inst = ELMo_Utils(args.elmo_vocab_file, args.elmo_weight_file, args.elmo_option_file, args.use_character_elmo, args.use_concat_p, args.question_window, args.elmo_u_cache_file, args.elmo_p_cache_file, args.elmo_q_cache_file)
    else:
        elmo_inst = None

    # get dialouge dataset instance
    dial_data = DialogueDataset(
        vocab,
        vocab_char,
        args,
        train_files=args.train_files,
        dev_files=args.dev_files,
        elmo_utils=elmo_inst
    )

    logger.info('Initialize the model...')
    dial_eval = DialogueEval(args, 'train.predicted')
    dial_model = DialogueModel(vocab, vocab_char, args, elmo_inst)

    if args.restore:
        logger.info('Restoring the model and continue training...')
        model_to_restore = args.model_prefix_to_restore if args.model_prefix_to_restore else args.algo
        dial_model.restore(model_dir=args.model_dir, model_prefix=model_to_restore)
        #dial_model.restore(model_dir=args.model_dir, model_prefix=args.algo)

    logger.info('Training the model...')
    dial_model.train(
        dial_data,
        args.epochs,
        args.batch_size,
        dial_eval,
        save_dir=args.model_dir,
        save_prefix=args.algo,
        evaluate=True)

    if elmo_inst:
        elmo_inst.cleanup()

    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("dial")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    if args.use_char_embedding:
        with open(os.path.join(args.vocab_dir, 'vocab_char.data'), 'rb') as fin:
            vocab_char = pickle.load(fin)
    else:
        vocab_char = None

    if len(args.elmo_positions):
        logger.info('Initializing ELMo Utils ...')
        elmo_inst = ELMo_Utils(args.elmo_vocab_file, args.elmo_weight_file, args.elmo_option_file, args.use_character_elmo, args.use_concat_p, args.question_window, args.elmo_u_cache_file, args.elmo_p_cache_file, args.elmo_q_cache_file)
    else:
        elmo_inst = None

    dial_data = DialogueDataset(
        vocab,
        vocab_char,
        args,
        dev_files=args.dev_files,
        elmo_utils=elmo_inst)

    logger.info('Restoring the model...')
    dial_model = DialogueModel(vocab, vocab_char, args, elmo_utils = elmo_inst)
    model_to_restore = args.model_prefix_to_restore if args.model_prefix_to_restore else args.algo
    dial_model.restore(model_dir=args.model_dir, model_prefix=model_to_restore)
    logger.info('Evaluating the model on dev set...')
    dev_batches = dial_data.gen_mini_batches(
        'dev', args.batch_size,
        shuffle=False)
    dial_eval = DialogueEval(args, result_prefix='dev.predicted.{}'.format(model_to_restore))
    dev_loss, metrics = dial_model.evaluate(
        dial_data.dev_context,
        dev_batches,
        dial_eval)
    logger.info('Loss on dev set: {}'.format(dev_loss))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("dial")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    if args.use_char_embedding:
        with open(os.path.join(args.vocab_dir, 'vocab_char.data'), 'rb') as fin:
            vocab_char = pickle.load(fin)
    else:
        vocab_char = None
    assert len(args.test_files) > 0, 'No test files are provided.'

    if len(args.elmo_positions):
        logger.info('Initializing ELMo Utils ...')
        elmo_inst = ELMo_Utils(args.elmo_vocab_file, args.elmo_weight_file, args.elmo_option_file, args.use_character_elmo, args.use_concat_p, args.question_window, args.elmo_u_cache_file, args.elmo_p_cache_file, args.elmo_q_cache_file)
    else:
        elmo_inst = None

    dial_data = DialogueDataset(
        vocab,
        vocab_char,
        args,
        test_files=args.test_files,
        elmo_utils=elmo_inst)

    logger.info('Restoring the model...')
    dial_model = DialogueModel(vocab, vocab_char, args, elmo_utils= elmo_inst)
    model_to_restore = args.model_prefix_to_restore if args.model_prefix_to_restore else args.algo
    dial_model.restore(model_dir=args.model_dir, model_prefix=model_to_restore)
    logger.info('Predicting answers for test set...')
    test_batches = dial_data.gen_mini_batches(
        'test',
        args.batch_size,
        shuffle=False)
    dial_eval = DialogueEval(args, 'test.predicted')
    dial_model.predict(
        dial_data.test_context,
        test_batches,
        dial_eval
    )
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))

def pipe_eval(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("dial")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    if args.use_char_embedding:
        with open(os.path.join(args.vocab_dir, 'vocab_char.data'), 'rb') as fin:
            vocab_char = pickle.load(fin)
    else:
        vocab_char = None

    if len(args.elmo_positions):
        logger.info('Initializing ELMo Utils ...')
        elmo_inst = ELMo_Utils(args.elmo_vocab_file, args.elmo_weight_file, args.elmo_option_file, args.use_character_elmo, args.use_concat_p, args.question_window, args.elmo_u_cache_file, args.elmo_p_cache_file, args.elmo_q_cache_file)
    else:
        elmo_inst = None

    dial_data = DialogueDataset(
        vocab,
        vocab_char,
        args,
        dev_files=args.dev_files,
        elmo_utils=elmo_inst)

    logger.info('Restoring the sub models...')
    # create pipeline dialogue model, which can be used to annotate the input utterance one by one according to their speakers.
    dial_model_pipe = DialogueModelPipe(vocab, vocab_char, args, elmo_utils = elmo_inst)
    dev_batches = dial_data.gen_mini_batches(
        'dev', args.batch_size,
        shuffle=False)
    dial_eval = DialogueEval(args, result_prefix='dev.predicted.{}'.format('pipe'))
    dev_loss, metrics = dial_model_pipe.evaluate(
        dial_data.dev_context,
        dev_batches,
        dial_eval)
    logger.info('Loss on dev set: {}'.format(dev_loss))

def ensemble(args):
    dial_eval = DialogueEval(args,'ensemble.predicted')
    if args.ensemble_method == 'RANK_SUM':
        predicted_samples = dial_eval.ensemble_by_ranksum(args.ensemble_predicted_files)
    elif args.ensemble_method == 'VOTE_TOP':
        predicted_samples = dial_eval.ensemble_by_votetop(args.ensemble_predicted_files)
    else:
        raise NotImplemented("ensemble method {} is not implemented yet.", args.ensemble_method)
    dial_eval.eval_metrics(predicted_samples)
    dial_eval.save_predicts(predicted_samples)

def run():
    """
    Prepares and runs the whole system.
    """
    args = config_reader.get_parser().parse_args()

    logger = logging.getLogger("dial")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    # Some GPU scheduler will broken by specifiying the CUDA_VISIBLE_DEVICES, if you need this, just uncomment it.
    # use 1,2,3 to set multiple gpuids

    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logger.warn("CUDA_VISIBLE_DEVICES speificed with {}".format(args.gpu))

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        if args.pipe:
            pipe_eval(args)
        else:
            evaluate(args)
    if args.predict:
        predict(args)

    if args.ensemble:
        ensemble(args)


if __name__ == '__main__':
    run()
