# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 14:41:38 jcao>
# --------------------------------------------------------------------
# File Name          : psyc_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : Tools for transforming the psyctherapy dataset
# --------------------------------------------------------------------

import json
import os, sys
import pandas as pd
import logging
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from collections import Counter
import random
# from datasketch import MinHash, MinHashLSH
# from nltk import ngrams

reload(sys)
sys.setdefaultencoding('utf8')

class PsycUtils(object):

    PADDING_SPEAKER = 'PAD'
    PADDING_LABEL = 'PADDING_LABEL'
    P_EXCEPTION_LABEL = 'P_EXCEPTION_LABEL'
    T_EXCEPTION_LABEL = 'T_EXCEPTION_LABEL'
    EXCEPTION_LABELS = [P_EXCEPTION_LABEL, T_EXCEPTION_LABEL]
    SPEAKERS = ['P','T']
    num_speaker = len(SPEAKERS)


    # brownie is used for a different task. Not used for this task.
    Other_labels = ['brownie']

    # original 9 P labels, on FN (Follow.Neutral), one neutral, '+' means positive 'changetalk'
    # '-' means negative 'sustain'
    MISC28_P_labels = ['follow_neutral','C+','C-','R+','R-','O+', 'O-', 'TS+', 'TS-']
    MISC28_BRIEF_P_labels = ['follow_neutral','C+','C-','R+','R-','O+', 'O-', 'TS+', 'TS-']
    # 3 labels for P, 126728 rows
    # noise labels for P
    # ['NC', 'NONE', 'GI', 'RES', 'FA', 'QUO']
    # For NC,  2327 rows, most of theirs utterances are [du], short setence, without much meansning, or noise information, just mark it FN.
    # For NONE, 22/126728 rows, most of them are not a complete sentence, treat them as neutral is fine, Some may also can be merged into a longer sentence.
    # For now, just ignore all NC and NONE labels.
    # For QUO, 3 rows, manually fixed it is wrong for the speaker.
    # For others , treat it as FN will be fine.

    # [batch_size, 2]
    MISC15_P_labels = ['change_talk','sustain_talk','follow_neutral']
    MISC15_BRIEF_P_labels = ['change_talk','sustain_talk','follow_neutral']

    MISC11_P_labels = ['change_talk','sustain_talk','follow_neutral']
    MISC11_BRIEF_P_labels = ['POS','NEG','FN']

    # labels in the annotated dataset
    # ['QUC', 'AF', 'ST', 'GI', 'RES', 'NC', 'QUO', 'FA', 'SU', 'ADW', 'REC', 'FI', 'EC', 'CO', 'ADP', 'NONE', 'DI', 'RCW', 'WA', 'RF', 'RCP']
    # for t, 141345 rows
    # NONE(21 rows ) and NC(2168 rows, 378 unique utterances) are two wrong labels.
    # NC and NONE are usually some interruption, or without any meaning.
    # I think we can just ignore it.
    # 19 labels for original T labels.
    MISC28_T_labels = ['reflection_complex','reflection_simple','affirm','facilitate','giving_info','question_open','question_closed','advise_wp','advise_wop','confront','structure','reframe','emphasize_control','raiseconcern_wp','raiseconcern_wop','warn','support', 'direct','filler']
    MISC28_BRIEF_T_labels = ['reflection_complex','reflection_simple','affirm','facilitate','giving_info','question_open','question_closed','advise_wp','advise_wop','confront','structure','reframe','emphasize_control','raiseconcern_wp','raiseconcern_wop','warn','support', 'direct','filler']

    # 12 T labels, 3 P labels. For original T in Mike's version.
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4842096/
    # MISC 15, it merged ['reframe','emphasize_control','raiseconcern_wp','raiseconcern_wop','warn','support', 'direct','filler']
    MISC15_T_labels = ['reflection_complex','reflection_simple','affirm','facilitate','giving_info','question_open','question_closed','advise_wp','advise_wop','confront','structure','other']
    MISC15_BRIEF_T_labels = ['reflection_complex','reflection_simple','affirm','facilitate','giving_info','question_open','question_closed','advise_wp','advise_wop','confront','structure','other']
    #T_labels = ['reflection_complex','reflection_simple','affirm','facilitate','givinginfo','question_open','question_closed','advise_wp','advise_wop','confront','structure','other']

    # MISC 11, http://scuba.usc.edu/pdf/xiao2016_behavioral-codi.pdf
    # 8 labels for T
    # MIA(MI adherent, merged 8 into 1): Affirm; Reframe; Emphasize control; Support; Filler; Advice with permission; Structure; Raise concern with permission
    # MIN(MI non adherent, merged 5 into 1) : Confront; Direct; Advice without permission; Warn; Raise concern without permission
    MISC11_T_labels = ['facilitate','reflection_simple','reflection_complex','giving_info','question_closed','question_open','MI_adherent','MI_non-adherent']

    MISC11_BRIEF_T_labels = ['FA','RES','REC','GI','QUC','QUO','MIA','MIN']

    PADDING_U_ID = "(-1)_-1_-1_PAD_-1_-1"
    PADDING_U_SAMPLE = {"turn_number": -1, "code_number": -1, "agg_label": PADDING_LABEL, "codes": [], "uid": PADDING_U_ID, "end_index": -1, "speaker": "PAD", "utterance": "", "start_index": -1}

    def __init__(self, cluster_strategy):
        self.logger = logging.getLogger("transform")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.cluster_strategy = cluster_strategy
        # In total 28 labels
        # All_labels = origin_P_labels + origin_T_labels
        # In total 8+3 = 11 lables.
        if 'MISC28' in self.cluster_strategy:
            self.P_labels = PsycUtils.MISC28_P_labels
            self.brief_P_labels = PsycUtils.MISC28_BRIEF_P_labels
            self.T_labels = PsycUtils.MISC28_T_labels
            self.breif_T_labels = PsycUtils.MISC28_BRIEF_T_labels
            self.translate_code_func = self.translate_code_MISC28_func
        elif 'MISC15' in self.cluster_strategy:
            self.P_labels = PsycUtils.MISC15_P_labels
            self.brief_P_labels = PsycUtils.MISC15_BRIEF_P_labels
            self.T_labels = PsycUtils.MISC15_T_labels
            self.brief_T_labels = PsycUtils.MISC15_BRIEF_T_labels
            self.translate_code_func = self.translate_code_MISC15_func
        elif 'MISC11' in self.cluster_strategy:
            self.P_labels = PsycUtils.MISC11_P_labels
            self.brief_P_labels = PsycUtils.MISC11_BRIEF_P_labels
            self.T_labels = PsycUtils.MISC11_T_labels
            self.brief_T_labels = PsycUtils.MISC11_BRIEF_T_labels
            self.translate_code_func = self.translate_code_MISC11_func
        else:
            raise NotImplementError("{} is not supported.".format(self.cluster_strategy))

        # by default, aggregate those unknown label into some known label, not using EXCEPTION_LABELS
        if 'WOE' in cluster_strategy:
            self.drop_exception = True
        elif 'EL' in cluster_strategy:
            self.drop_exception = False
            self.P_labels += [PsycUtils.P_EXCEPTION_LABEL]
            self.T_labels += [PsycUtils.T_EXCEPTION_LABEL]
        elif 'ML' in cluster_strategy:
            # just use the majority label
            self.drop_exception = True

        self.All_labels = self.P_labels + self.T_labels
        self.brief_All_labels = self.brief_P_labels + self.brief_T_labels
        self.num_all_labels = len(self.All_labels)
        self.All_labels_with_padding = self.All_labels + [PsycUtils.PADDING_LABEL]
        self.num_all_labels_with_padding = len(self.All_labels_with_padding)
        self.All_speakers = PsycUtils.SPEAKERS
        self.num_all_speakers = len(PsycUtils.SPEAKERS)
        self.All_speakers_with_padding = self.All_speakers + [PsycUtils.PADDING_SPEAKER]
        self.num_all_speakers_with_padding = len(self.All_speakers_with_padding)

    def translate_code_MISC28_func(self, code, speaker):
        # to lower the coder, just in case the code is mixed with upper and lower case.
        lower_code = code.lower()
        # brownie is for a different annotating suppose. But there are still some codes with only brownie codes.
        if ':-)' in lower_code:
            # To be fixed
            return 'brownie'
        elif speaker == 'P':
            if 'c+' in lower_code:
                return 'C+'
            elif 'c-' in lower_code:
                return 'C-'
            if 'r+' in lower_code:
                return 'R+'
            elif 'r-' in lower_code:
                return 'R-'
            if 'o+' in lower_code:
                return 'O+'
            elif 'o-' in lower_code:
                return 'O-'
            if 'ts+' in lower_code:
                return 'TS+'
            elif 'ts-' in lower_code:
                return 'TS-'
            elif 'fn' == lower_code:
                return 'follow_neutral'
            else:
                #return 'exception'
                return 'follow_neutral'
        elif speaker == 'T':
            if 'rec' == lower_code:
                return 'reflection_complex'
            elif 'res' == lower_code:
                return 'reflection_simple'
            elif 'fa' == lower_code:
                return 'facilitate'
            elif 'gi' == lower_code:
                return 'giving_info'
            elif 'quo' == lower_code:
                return 'question_open'
            elif 'quc' == lower_code:
                return 'question_closed'
            elif 'af' == lower_code:
                return 'affirm'
            elif 'rf' == lower_code:
                return 'reframe'
            elif 'ec' == lower_code:
                return 'emphasize_control'
            elif 'su' == lower_code:
                return 'support'
            elif 'fi' == lower_code:
                return 'filler'
            elif 'adp' == lower_code:
                return 'advise_wp'
            elif 'st' == lower_code:
                return 'structure'
            elif 'rcp' == lower_code:
                return 'raiseconcern_wp'
            elif 'co' == lower_code:
                return 'confront'
            elif 'di' == lower_code:
                return 'direct'
            elif 'adw' == lower_code:
                return 'advise_wop'
            elif 'wa' == lower_code:
                return 'warn'
            elif 'rcw' == lower_code:
                return 'raiseconcern_wop'
            else:
                # 'reframe','emphasize_control','raiseconcern_wp','raiseconcern_wop','warn','raiseconcern_wop', 'direct','filler']A
                # merged into other
                # self.logger.warn("Unsupported code type = {} when speaker = {}".format(code, speaker))
                #return 'exception'
                return 'facilitate'
        else:
            # self.logger.warn("Unsupported speaker type = {}".format(code, speaker))
            return 'exception'
        return ''

    def translate_code_MISC11_func(self, code, speaker):
        # to lower the coder, just in case the code is mixed with upper and lower case.
        lower_code = code.lower()
        # brownie is for a different annotating suppose. But there are still some codes with only brownie codes.
        if ':-)' in lower_code:
            # To be fixed
            return 'brownie'
        elif speaker == 'P':
            if '+' in lower_code:
                return 'change_talk'
            elif '-' in lower_code:
                return 'sustain_talk'
            else:
                return 'follow_neutral'
        elif speaker == 'T':
            if 'rec' == lower_code:
                return 'reflection_complex'
            elif 'res' == lower_code:
                return 'reflection_simple'
            elif 'fa' == lower_code:
                return 'facilitate'
            elif 'gi' == lower_code:
                return 'giving_info'
            elif 'quo' == lower_code:
                return 'question_open'
            elif 'quc' == lower_code:
                return 'question_closed'
            elif lower_code in ['af','rf','ec','su','fi','adp','st','rcp']:
                return 'MI_adherent'
            elif lower_code in ['co','di','adw','wa','rcw']:
                return 'MI_non-adherent'
            else:
                # 'reframe','emphasize_control','raiseconcern_wp','raiseconcern_wop','warn','raiseconcern_wop', 'direct','filler']A
                # merged into other
                # self.logger.warn("Unsupported code type = {} when speaker = {}".format(code, speaker))
                return 'facilitate'
                #return 'exception'
        else:
            # self.logger.warn("Unsupported speaker type = {}".format(code, speaker))
            return 'exception'
        return ''

    # MISC 15, 19 original codes, 11 from the original, the rest will merge into a single other. Actually, in Mike's version, there are 21 codes, 2 of them are not correct, they will put into 'other'.
    def translate_code_MISC15_func(self, code, speaker):
        lower_code = code.lower()
        # brownie is for a different annotating suppose. But there are still some codes with only brownie codes.
        if ':-)' in lower_code:
            return 'brownie'
        elif speaker == 'P':
            if '+' in lower_code:
                return 'change_talk'
            elif '-' in lower_code:
                return 'sustain_talk'
            else:
                return 'neutral'
        elif speaker == 'T':
            if 'rec' == lower_code:
                return 'reflection_complex'
            elif 'res' == lower_code:
                return 'reflection_simple'
            elif 'af' == lower_code:
                return 'affirm'
            elif 'fn' == lower_code or 'fa' in lower_code:
                return 'facilitate'
            elif 'gi' == lower_code:
                return 'giving_info'
            elif 'quo' == lower_code:
                return 'question_open'
            elif 'quc' == lower_code:
                return 'question_closed'
            elif 'adp' == lower_code:
                return 'advise_wp'
            elif 'adw' == lower_code:
                return 'advise_wop'
            elif 'co' == lower_code:
                return 'confront'
            elif 'st' == lower_code:
                return 'structure'
            else:
                # 'reframe','emphasize_control','raiseconcern_wp','raiseconcern_wop','warn','raiseconcern_wop', 'direct','filler']
                # merged into other
                # self.logger.warn("Unsupported code type = {} when speaker = {}".format(code, speaker))
                return 'other'
        else:
            # self.logger.warn("Unsupported speaker type = {}".format(code, speaker))
            return 'other'
        return ''

    def _create_turn_sample(self, sessionid, turn_number, t_rows):
        turn_sample = {
            'turn_numbder' : turn_number,
            'utterance_samples' : []
        }
        # sort the utterance in the one turn given that speaker.
        # every utterance can be uniquely by its turn_number, speaker, start_index, end_index
        speaker_start_end_gps = t_rows.groupby(['speaker', 'start_index','end_index'])
        # different utterance will have different start and end index.
        # To sort those utterance in a turn, we should consider their start and end index, to makesure it is sorted and integrated.
        for (speaker, start_index, end_index), u_rows in speaker_start_end_gps:
            u_sample = self._create_utterance_sample(sessionid, turn_number, speaker, start_index, end_index, u_rows)
            turn_sample['utterance_samples'].append(u_sample)
        turn_sample['utterance_samples'].sort(key=lambda u: u['code_number'])
        return turn_sample


    def _create_utterance_sample(self, sessionid, turn_number, speaker, start_index, end_index, u_rows):
        utterance_sample = {}
        # set speaker
        utterance_sample['speaker'] = speaker
        # set start_index
        utterance_sample['start_index'] = start_index
        # set end_index
        utterance_sample['end_index'] = end_index
        utterance_sample['turn_number'] = turn_number
        # set the utterance
        unique_utterances = u_rows.utterance.unique()
        if len(unique_utterances) == 0:
            utterance_sample['utterance'] = ""
            # self.logger.warn("same start and end index, but different utterance, {}".format(unique_utterances))
        else:
            # only unique utterance for that start_index and end_index
            utterance_sample['utterance'] = unique_utterances[0]

        # set the coder number
        unique_code_numbers = u_rows.code_number.unique()
        utterance_sample['code_number'] = unique_code_numbers[0]
        # set uid
        utterance_sample['uid'] = '({})_{}_{}_{}_{}_{}'.format(sessionid, turn_number, unique_code_numbers[0], speaker, start_index, end_index)

        # set coderid, order_id, code_number, code
        utterance_sample['codes'] = []
        # for patients, codes : all:521, changetalk:316, sustain:198, other:7
        # for therapy,  codes : 40,
        #['QUC' ':-)E' 'AF' 'ST' 'GI' ':-)AS' ':-)C' 'RES' 'QUO' ':-)EV' 'FA' 'SU'
        # ':-)A' 'ADW' 'REC' 'NC' ':-)S' 'FI' 'EC' 'CO' 'ADP' 'DI' 'NONE' 'RCW'
        # ':-)FEEL' ':-)AT' ':-)CF' ':-)SK' ':-)WM' ':-)CN' ':-)COG' ':-)FR'
        # ':-)UN' ':-)AC' 'WA' ':-)EX' 'RF' ':-)RE' 'RCP' 'FN']
        # 18 :-) , 22
        code_gps = u_rows.groupby(['code'])
        for code, c_rows in code_gps:
            code_sample = self._create_code_smaple(code, speaker, c_rows)
            utterance_sample['codes'].append(code_sample)

        return utterance_sample

    def _create_code_smaple(self, code, speaker, c_rows):
        code_sample = {
            'origin_code': code,
            'translated_code' : self.translate_code_func(code, speaker),
            # code_num seems uselss.
            'coder_order': []
        }
        for index, row in c_rows.iterrows():
            code_order_sample = {
                'coder_id' : row['coderid'],
                'order_id' : row['order_id'],
                'cid' : row['id']
            }
        code_sample['coder_order'].append(code_order_sample)
        return code_sample

    def transform(self, ori_file, json_file):
        df = pd.read_csv(ori_file, sep='|')
        sess_gps = df.groupby(by=['sessionid'])
        all_sess_samples = []
        all_sess_norm_ids = []
        for sessionid, rows in tqdm(sess_gps):
            sess_sample = {}
            # only count the first 2 part after splitting with "_"
            norm_session_id = '_'.join(sessionid.split('_')[:2])
            if norm_session_id in all_sess_norm_ids:
                continue

            sess_sample['sessionid'] = norm_session_id
            sess_sample['isholdout'] = rows.isholdout.unique()[0]
            sess_sample['turn_samples'] = []
            sp_turn_gps = rows.groupby(['turn_number'])
            # it will keep the original utterance order
            for turn_number, t_rows in sp_turn_gps:
                turn_sample = self._create_turn_sample(norm_session_id, turn_number, t_rows)
                sess_sample['turn_samples'].append(turn_sample)

            # add sess_sample
            all_sess_samples.append(sess_sample)
            all_sess_norm_ids.append(norm_session_id)

        with open(json_file, 'w') as outfile:
            json.dump(all_sess_samples, outfile, ensure_ascii=False, indent=4)

    def get_most_common_label(self, utterance):
        # just use the agg_label if existed
        if 'agg_label' in utterance:
            return utterance['agg_label']

        if utterance['speaker'] == 'T':
            correct_labels = [c['translated_code'] for c in utterance['codes'] if c['translated_code'] in self.T_labels]
            if 'ML' in self.cluster_strategy:
                exception_label = 'facilitate'
            else:
                exception_label = PsycUtils.T_EXCEPTION_LABEL
        elif utterance['speaker'] == 'P':
            correct_labels = [c['translated_code'] for c in utterance['codes'] if c['translated_code'] in self.P_labels]
            if 'ML' in self.cluster_strategy:
                exception_label = 'follow_neutral'
            else:
                exception_label = PsycUtils.P_EXCEPTION_LABEL
        else:
            # for padding utterance and padding speaker, just return padding label
            correct_labels = []
            exception_label = PsycUtils.PADDING_LABEL

        if not correct_labels:
            # if only brownie label in it, treat it as facilitate
            # empty labels:
            # self.logger.info("not label for utterance = {}".format(utterance))
            return exception_label
        else:
            label_counter = Counter(correct_labels)
            most_common_freq = label_counter.most_common(1)[0][1]
            # For the tied common label, choose the majority first, then random choose one.
            keys = [ key for (key, cnt) in label_counter.items() if cnt == most_common_freq ]
            # randomly choose one, if tied
            most_common_label = np.random.choice(keys)
            return most_common_label

    def seg_dial_by_u_window(self, full_dial_file, output_folder, utterance_window, dev_id_file = None, dev_portion = 0.1, padding_ahead = True, padding_end = False, drop_exception = True):
        statistics = {
                'train' : { "labels" : {"total": 0}, "last_label": {"total": 0 }, "total_sess": 0, "total_u":0, "total_dup": 0, "total_example" : 0, "P2T" : 0, "P2P":0, "T2P": 0, "T2T": 0},
                'dev' : { "labels" : {"total": 0}, "last_label": {"total" : 0}, "total_sess" : 0, "total_u" : 0, "total_dup": 0, "total_example": 0, "P2T" : 0, "P2P":0, "T2P": 0, "T2T": 0},
                'test' : { "labels" : {"total": 0}, "last_label": {"total" : 0}, "total_sess" : 0, "total_u" : 0, "total_dup": 0, "total_example": 0, "P2T" : 0, "P2P":0, "T2P": 0, "T2T": 0}
        }

        for label in self.All_labels + [PsycUtils.PADDING_LABEL]:
            statistics['train']['labels'][label] = 0
            statistics['dev']['labels'][label] = 0
            statistics['test']['labels'][label] = 0
            statistics['train']['last_label'][label] = 0
            statistics['dev']['last_label'][label] = 0
            statistics['test']['last_label'][label] = 0

        # build file paths
        padding_suffix = "_padding" if padding_ahead else ""
        train_file = os.path.join(output_folder, "train.json")
        dev_file = os.path.join(output_folder, "dev.json")
        test_file = os.path.join(output_folder, "test.json")
        complete_file = os.path.join(output_folder, "complete.json")
        stat_file = os.path.join(output_folder, "stat.json")

        specified_dev_ids = []
        if dev_id_file and os.path.exists(dev_id_file):
            self.logger.info("load dev_id_file from {}".format(dev_id_file))
            with open(dev_id_file, 'r') as devid_fin:
                for dev_id in devid_fin:
                    specified_dev_ids.append(dev_id.rstrip('\n'))
                self.logger.info('dev ids are {}'.format(specified_dev_ids))

        with open(full_dial_file, 'r') as fin, open(train_file,'w') as ftrain, open(dev_file, 'w') as fdev, open(test_file, 'w') as ftest, open(complete_file,'w') as f_complete:
            sess_samples = json.load(fin)
            total_sample = len(sess_samples)
            total_holdout = len([s for s in sess_samples if s['isholdout'] == 't'])
            total_train = total_sample - total_holdout
            fdict = {
                'train': ftrain,
                'dev' : fdev,
                'test' : ftest
            }
            complete_dials = []
            session_keys = {}
            # session_minhashs = {}
            # similarity_threshold = 0.9
            # lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)
            # for sample_index in tqdm(range(10)):
            for sample_index in tqdm(np.random.permutation(range(total_sample))):
                # for every sess_samples, get its turn samples, then get all the continuous combination of #window turns.
                sess_sample = sess_samples[sample_index]

                # make statistics for the whole session.
                statistics_in_session = {"labels":{"total": 0}, "last_label": {"total" : 0}, "total_dup": 0}
                for label in self.All_labels + [PsycUtils.PADDING_LABEL]:
                    statistics_in_session['labels'][label] = 0
                    statistics_in_session['last_label'][label] = 0

                turn_samples = sess_sample['turn_samples']
                turn_length = len(turn_samples)
                # generate all utterance seq given turn length
                possible_utterance_seqs = self.gen_turn_seq(turn_samples, 0, turn_length, drop_exception)
                if len(possible_utterance_seqs) > 1:
                    self.logger.info("sessionid = {}, variants = {}".format(sess_sample['sessionid'], len(possible_utterance_seqs)))

                max_len_seq = max(possible_utterance_seqs, key=lambda x: len(x))
                s_turn = max_len_seq[0]['turn_number']
                s_speaker = max_len_seq[0]['speaker']
                s_sindex = max_len_seq[0]['start_index']
                s_eindex = max_len_seq[0]['end_index']
                e_turn = max_len_seq[-1]['turn_number']
                e_speaker = max_len_seq[-1]['speaker']
                e_sindex = max_len_seq[-1]['start_index']
                e_eindex = max_len_seq[-1]['end_index']
                #concat_utterance = reduce(lambda x,y: x+' '+y, map(lambda s: s['utterance'], max_len_seq))
                #minhash = MinHash(num_perm=128)
                #for d in ngrams(concat_utterance, 3):
                #    minhash.update("".join(d).encode('utf-8'))
                #lsh.insert(sess_sample['sessionid'], minhash)
                #session_minhashs[sess_sample['sessionid']] = minhash

                session_key = "{}_({}_{}_{}_{})-({}_{}_{}_{})".format(len(max_len_seq), s_turn, s_speaker, s_sindex, s_eindex, e_turn, e_speaker, e_sindex, e_eindex)
                if session_key in session_keys:
                    session_keys[session_key]["sessionids"].append(sess_sample['sessionid'])
                    session_keys[session_key]["length"] += 1
                else:
                    session_keys[session_key] = {"sessionids":[], "length":0}
                    session_keys[session_key]["sessionids"].append(sess_sample['sessionid'])
                    session_keys[session_key]["length"] += 1

                example_ids_set = set()
                dial_examples_in_session = []
                for possible_seq in [max_len_seq]:
                #for possible_seq in possible_utterance_seqs:
                    # for all possible utterance seq for whole conversation:
                    if not padding_end:
                        seq_length = len(possible_seq)
                    else:
                        # padding n-1 pad sample at the end of sequence.
                        possible_seq = possible_seq + (utterance_window - 1) * [ PsycUtils.PADDING_U_SAMPLE ]
                        seq_length = len(possible_seq)

                    for u in possible_seq:
                        most_common_label = self.get_most_common_label(u)
                        u['agg_label'] = most_common_label

                    for i in range(0, seq_length):
                        # i is the end of a conversation, when they nor form a whole window, padding them head of i
                        # a window is unique for <start_turn, start_not index> <end_turn, end_index>
                        start = i + 1 - utterance_window
                        if start >=0:
                            u_seqs = possible_seq[start:i+1]
                            start_turn = u_seqs[0]['turn_number']
                            start_speaker = u_seqs[0]['speaker']
                            start_s_index = u_seqs[0]['start_index']
                            end_turn = u_seqs[-1]['turn_number']
                            end_speaker = u_seqs[-1]['speaker']
                            end_e_index = u_seqs[-1]['end_index']
                        else:
                            if not padding_ahead:
                                # if not padding ahead, then this will not form a example
                                continue
                            else:
                                # add padding ahead of the start
                                u_seqs =  (0 - start) * [ PsycUtils.PADDING_U_SAMPLE ] + possible_seq[:i+1]
                                start_turn = u_seqs[0]['turn_number']
                                start_speaker = u_seqs[0]['speaker']
                                start_s_index = u_seqs[0]['start_index']
                                end_turn = u_seqs[-1]['turn_number']
                                end_speaker = u_seqs[-1]['speaker']
                                end_e_index = u_seqs[-1]['end_index']
                        example_id = '({})_({}, {}, {})-({}, {}, {})'.format(sess_sample['sessionid'], start_speaker, start_turn, start_s_index, end_speaker, end_turn, end_e_index)
                        if example_id in example_ids_set:
                            self.logger.info("duplicate example_id = {} ".format(example_id))
                            statistics_in_session['total_dup'] += 1
                            continue
                        else:
                            example_ids_set.add(example_id)
                            dial_example = {}
                            dial_example['example-id'] = example_id
                            dial_example['messages-so-far'] = u_seqs[:-1]
                            # make all codes as corres codes
                            dial_example['options-for-correct-answers'] = [u_seqs[-1]]
                            # make label statistics as current label for dial example
                            for u in u_seqs:
                                agg_label = u['agg_label']
                                statistics_in_session['labels'][agg_label] += 1
                                statistics_in_session['labels']['total'] += 1

                            most_common_label = u_seqs[-1]['agg_label']
                            statistics_in_session['last_label'][most_common_label] += 1
                            statistics_in_session['last_label']['total'] += 1

                            dial_examples_in_session.append(dial_example)

                if sess_sample['isholdout'] == 't':
                    data_split_name = 'test'
                else:

                    if specified_dev_ids:
                        if sess_sample['sessionid'] in specified_dev_ids:
                            self.logger.info('{} is in dev_dis'.format(sess_sample['sessionid']))
                            data_split_name = 'dev'
                        else:
                            data_split_name = 'train'
                    else:
                        if statistics['dev']['total_sess'] < total_train * dev_portion:
                            # where dev lacks, but the current uttt statuserance helps
                            lacking_labels = [ x for x in self.All_labels if statistics['dev']['last_label'][x] == 0 ]
                            complement_labels = [ x for x in lacking_labels if statistics_in_session['last_label'][x] > 0]
                            if lacking_labels:
                                if complement_labels:
                                    data_split_name = 'dev'
                                else:
                                    # no complement labels for lacking, make it train, wait for next dev
                                    data_split_name = 'train'
                            else:
                                # not lacking, just get it dev until the dev is full
                                data_split_name = 'dev'
                        else:
                            # if dev is already ready, make it into train
                            data_split_name = 'train'

                complete_dial = {}
                complete_dial['sessionid'] = sess_sample['sessionid']
                complete_dial['split_name'] = data_split_name
                complete_dial['possible_utterances'] = possible_utterance_seqs
                complete_dials.append(complete_dial)
                # add label statistics only for the last utterance
                # as we slide window by 1, then when we padding all of them, #examples == #utterance
                for k in self.All_labels + [PsycUtils.PADDING_LABEL]:
                    statistics[data_split_name]['labels'][k] += statistics_in_session['labels'][k]
                    statistics[data_split_name]['last_label'][k] += statistics_in_session['last_label'][k]

                # add total
                statistics[data_split_name]['labels']['total'] += statistics_in_session['labels']['total']
                statistics[data_split_name]['last_label']['total'] += statistics_in_session['last_label']['total']
                statistics[data_split_name]['total_dup'] += statistics_in_session['total_dup']
                # add session and utterance statistics
                statistics[data_split_name]["total_sess"] += 1
                statistics[data_split_name]["total_u"] += max([len(x) for x in possible_utterance_seqs])
                for e in dial_examples_in_session:
                    json.dump(e, fdict[data_split_name], encoding="utf-8")
                    fdict[data_split_name].write("\n")
                    last_speaker = e['messages-so-far'][-1]['speaker']
                    next_speaker = e['options-for-correct-answers'][0]['speaker']
                    statistics[data_split_name]['total_example'] += 1
                    sp2sp_str = '{}2{}'.format(last_speaker, next_speaker)
                    if sp2sp_str in statistics[data_split_name]:
                        statistics[data_split_name][sp2sp_str] += 1

            json.dump(complete_dials, f_complete, indent = 4, encoding="utf-8")

        # finally, print all the statistics
        self.logger.info("Statistics for by utterance window {} :\n {} \n {}, duplicate_session check keys: \n  {}".format(utterance_window, json.dumps(statistics, indent = 4, encoding="utf-8"), json.dumps(session_keys, indent=4, encoding="utf-8"), len(session_keys)))
        with open(stat_file, "w") as fstat:
            json.dump(statistics, fstat, indent = 4, encoding="utf-8")
            json.dump(session_keys, fstat, indent = 4, encoding="utf-8")
            # fstat.write("jacob similarity > {}: \n".format(similarity_threshold))
            # for i in session_minhashs.keys():
            #     result = lsh.query(session_minhashs[i])
            #     fstat.write("similary sessions for {}, sim_num = {}, {}\n".format(i, len(result), result))

    def gen_turn_seq(self, turn_samples, start_turn, window, drop_exception):
        possible_seq = []
        if window == 0 or start_turn >= len(turn_samples):
            # self.logger.info("start turn = {}, window={}".format(start_turn, window))
            return [[]]
        else:
            current_sorted_u_seqs=[]
            while True:
                if window == 0 or start_turn >= len(turn_samples):
                    # self.logger.info("current sorted_q_seqs in start turn = {}, window={} is {}".format(start_turn, window, len(current_sorted_u_seqs)))
                    return current_sorted_u_seqs
                u_samples_in_current_turns = turn_samples[start_turn]['utterance_samples']
                all_p_u_seqs = self.gen_utterance_seq('P',0, u_samples_in_current_turns, drop_exception)
                all_t_u_seqs = self.gen_utterance_seq('T',0, u_samples_in_current_turns, drop_exception)
                sorted_u_seqs = [sorted(p_u_seq+t_u_seq, key=lambda x: x['code_number']) for (p_u_seq, t_u_seq) in itertools.product(all_p_u_seqs, all_t_u_seqs)]
                start_turn = start_turn + 1
                window = window -1
                if len(sorted_u_seqs) == 1:
                    # not in stack, continue roll in
                    if len(current_sorted_u_seqs) == 1:
                        current_sorted_u_seqs[0] = current_sorted_u_seqs[0] + sorted_u_seqs[0]
                    else:
                        current_sorted_u_seqs.extend(sorted_u_seqs)
                else:
                    combine = []
                    if len(current_sorted_u_seqs):
                        pre = current_sorted_u_seqs[0]
                    else:
                        pre = []
                    for x in sorted_u_seqs:
                        combine.append(pre + x)
                    current_sorted_u_seqs = combine
                    break
            # self.logger.info("current sorted_q_seqs in start turn = {}, window={} is {}".format(start_turn, window, len(current_sorted_u_seqs)))
            tail_seqs = self.gen_turn_seq(turn_samples, start_turn, window, drop_exception)
            for u_seq in current_sorted_u_seqs:
                for tail_seq in tail_seqs:
                    possible_seq.append(u_seq + tail_seq)
            if len(possible_seq) == 0:
                possible_seq = [[]]
        return possible_seq

    def gen_utterance_seq(self, speaker, start_index, utterance_samples, drop_exception = True):
        """
        For now, given a start index, and speaker, the utterance in that turn, it will only return one possible seq.
        1. choose the utterance with max length
        2. if the most_common_label is not in EXCEPTION_LABELS, the not use it

        It can have multiple utterance seq, when there are different segementaion of utterance. That will make our data duplicate a lot.
        So we don't use those different segmenation now, just prepfer the longer sentence given a start_index.
        """
        all_u_seq = []
        current_utterances = [x for x in utterance_samples if x['speaker'] == speaker and x['start_index'] == start_index]
        if len(current_utterances) == 0:
            all_u_seq.append([])
        else:
            # if in the same turn , there are multiple utterance share the same start index, then choose the max end index
            u_with_max_end_index = max(current_utterances, key = lambda x: x['end_index'])
            # if current utterances are in exception code, then not use it
            possible_u_seqs = self.gen_utterance_seq(speaker, u_with_max_end_index['end_index'], utterance_samples, drop_exception)
            for p_u_seq in possible_u_seqs:
                # if current utterances are in exception code, then not use it
                most_common_label = self.get_most_common_label(u_with_max_end_index)
                if drop_exception and most_common_label in PsycUtils.EXCEPTION_LABELS:
                    all_u_seq.append([] + p_u_seq)
                else:
                    all_u_seq.append([u_with_max_end_index] + p_u_seq)
        return all_u_seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform_from', nargs='?', required=False, help='ori psyc data file')
    parser.add_argument('--cluster_strategy', choices=['MISC11_WOE','MISC15_WOE','MISC28_WOE','MISC11_EL','MISC15_EL','MISC28_EL','MISC11_ML','MISC15_ML','MISC28_ML'],required=True, help='how to cluster the MISC codes, EL means using exception label, ML means when exception, using majority label')
    parser.add_argument('--full_dial', nargs='?', required=False, help='the full dialouge json file to save')
    parser.add_argument('--seg_dial_by_utterance', nargs='?', type=int, required=False, help='generate dial segementaion data given number of utterance')
    parser.add_argument('--dev_id_file', nargs='?', required=False, help='dev id file, which can be load to determine the dev sessionids')
    parser.add_argument('--padding_ahead', required=False, action='store_true', help='whether padding ahead')
    parser.add_argument('--padding_end', required=False, action='store_true', help='whether padding end')
    parser.add_argument('--output_folder', nargs='?', required=False, help='the output folder for generating dial segementaion')

    args = parser.parse_args()
    if args.transform_from is not None and args.cluster_strategy is not None and args.full_dial is not None:
        # transfrom the ori file
        psyc_utils = PsycUtils(args.cluster_strategy)
        psyc_utils.transform(args.transform_from, args.full_dial)

    if args.full_dial is not None and args.seg_dial_by_utterance is not None:
        psyc_utils = PsycUtils(args.cluster_strategy)
        psyc_utils.seg_dial_by_u_window(args.full_dial, args.output_folder, utterance_window = args.seg_dial_by_utterance, dev_id_file = args.dev_id_file, dev_portion = 0.1, padding_ahead = args.padding_ahead, padding_end=args.padding_end, drop_exception = psyc_utils.drop_exception)
