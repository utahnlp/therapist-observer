# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 15:17:31 jcao>
# --------------------------------------------------------------------
# File Name          : preprocess_psyc_dataset.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A tool to read and preprocess the psyc dataset.
# --------------------------------------------------------------------


# -*- coding: utf-8 -* import sys
import sys
import os
import ujson as json
import operator
from tqdm import tqdm
from classes.snt_dict import SentenceDict
reload(sys)
sys.setdefaultencoding('utf8')


class preprocess_psyc_dataset(object):
    def __init__(self, input_json_files, output_folder, remove_stops=False, replace_tokens=False):
        self.remove_stops = True if remove_stops else False
        self.replace_tokens = replace_tokens if replace_tokens else False

        self.data_folder = output_folder

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        elmo_snt_dict_file = os.path.join(self.data_folder, 'elmo_snt_dict_file')
        # Attention Please!! make sure the padding sentence is the first one.
        self.snt_dict = SentenceDict(elmo_snt_dict_file, lower=False, init_snts = [''])

        for input_json_file in input_json_files:
            output_json_file = input_json_file.replace('.tokenized', '')
            file_name = os.path.basename(output_json_file)
            output_json_file = os.path.join(self.data_folder, file_name)
            # save tokenized json to a new json in preprocessed folder
            self.save_json(input_json_file, output_json_file)

        # dump the snt_dict into unique_snt_files, which can be loaded into snt_files again, and also it can be used to buid elmo embedding.
        print "save snt_dict to %s" % elmo_snt_dict_file
        self.snt_dict.dump_snts_file_with_start_end()
        self.snt_dict.dump_snts_file_without_start_end()

    def save_json(self, input_json_file, output_json_file):

        if os.path.exists(output_json_file):
            print "%s exists, skipping..." % output_json_file
        else:
            print "Processing: %s" % output_json_file
            with open(output_json_file, 'w') as f:
                for l in tqdm(open(input_json_file)):
                    tmp_dict = {}
                    j = json.loads(l)
                    tmp_dict['example-id'] = j['example-id']
                    tmp_dict['options-for-correct-answers'] = []
                    for option in j['options-for-correct-answers']:
                        opt = {}
                        opt['tokenized_utterance'] = self._get_tokens(option['tokenized_utterance'])
                        opt['agg_label'] = option['agg_label']
                        opt['codes'] = option['codes']
                        opt['uid'] = option['uid']
                        opt['speaker'] = option['speaker']
                        opt['snt_id'] = self.snt_dict.add(' '.join(opt['tokenized_utterance']))
                        tmp_dict['options-for-correct-answers'].append(opt)

                    tmp_dict['messages-so-far'] = []
                    for option in j['messages-so-far']:
                        opt = {}
                        opt['tokenized_utterance'] = self._get_tokens(option['tokenized_utterance'])
                        opt['agg_label'] = option['agg_label']
                        opt['codes'] = option['codes']
                        opt['uid'] = option['uid']
                        opt['speaker'] = option['speaker']
                        opt['snt_id'] = self.snt_dict.add(' '.join(opt['tokenized_utterance']))
                        tmp_dict['messages-so-far'].append(opt)

                    f.write(json.dumps(tmp_dict) + '\n')

    def get_data_dir(self):
        return self.data_folder

    def _get_tokens(self, tokens):
        if not tokens:
            return []
        else:
            # replace tokens
            tokens = reduce(operator.concat, map(self._replace_token, tokens))
            # remove stopwords
            if self.remove_stops:
                tokens = filter(self._is_not_stop, tokens)

            return [token['token'] for token in tokens]

    # replace tokens with title, expanded title or none
    def _replace_token(self, token):
        if token['type'] == 'normal_token':
            return [token]
        else:
            # TODO: handle other speciall entities.
            if self.replace_tokens:
                return [token]
            else:
                return [token]

    # returns true is a token is not a stop word
    def _is_not_stop(self, token):
        return not token['is_stop']
