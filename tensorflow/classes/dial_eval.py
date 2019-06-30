# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 14:57:36 jcao>
# --------------------------------------------------------------------
# File Name          : dial_eval.py
# Original Author    : jiessie.cao@gmail.com
# Description        : Evalution Tools for psyc dialogue
# --------------------------------------------------------------------

import logging
import math_utils
import os
import json
import copy
import numpy as np
from psyc_utils import PsycUtils 
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

class DialogueEval(object):
    """
    All the metrics to be evaluated
    """
    def __init__(self, args, result_prefix):
        self.logger = logging.getLogger("dial")
        self.topK_list = args.topK_list
        self.topM_for_acc_prob = args.topM_for_acc_prob
        self.acc_sum_prob = args.acc_sum_prob
        self.result_prefix = result_prefix
        self.result_dir = args.result_dir + "/" + self.result_prefix + "/"
        try:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
                self.logger.info("create result dir = {}".format(self.result_dir))
        except OSError as e:
            raise OSError("failed to create dirs {}".format(self.result_dir))

        self.decode_goal = args.decode_goal
        self.psyc_utils = PsycUtils(args.cluster_strategy)
        self.gen_confusing_matrix = True if args.evaluate else False

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


    def eval_metrics(self, predicted_answers):
        if 'SEQTAG' in self.decode_goal:
            return self.eval_metrics_for_seqtags(predicted_answers)
        else:
            return self.eval_metrics_for_multiclass(predicted_answers)

    def eval_metrics_for_seqtags(self, predicted_answers):
        """
        Assuming that the 'pred_seq_tags' has been assigned already
        """
        total_correct_in_all = 0
        label_pred = []
        label_true = []
        label_weights = []
        digits = 3
        metrics = {}

        for e_id, sample in predicted_answers.iteritems():
            # get all correct ids, include padding ids.
            correct_label_indices = sample['correct_seq_labels']
            # use extend to add all the labels in the seq, include the head padding and tail padding
            label_true.extend(correct_label_indices)
            # counting all correct for each sample
            total_correct_in_all += len(correct_label_indices)
            # select topK
            label_pred.extend(sample['pred_seq_tags'])

        if total_correct_in_all != 0:
            p, r, f1, s = precision_recall_fscore_support(label_true, label_pred, beta=1.0, labels=range(self.num_classes), average=None)
            total_s = np.sum(s)
            p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(label_true, label_pred, beta=1.0, labels=range(self.num_classes), average='micro')
            last_lines_heading = ['macro / total', 'weighted_mac / total', 'micro / total']
            target_names = self.classes
            name_width = max(len(cn) for cn in target_names)
            width = max(name_width, max([len(x) for x in last_lines_heading]), digits)

            headers = ["precision", "recall", "f1-score", "support"]
            head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
            report = head_fmt.format(u'', *headers, width=width)
            report += u'\n\n'
            row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
            rows = zip(target_names, p, r, f1, s)
            for row in rows:
                label_weights.append(row[4])
                report += row_fmt.format(*row, width=width, digits=digits)
                metrics['P_{}'.format(row[0])] = (1, row[1])
                metrics['R_{}'.format(row[0])] = (1, row[2])
                metrics['F1_{}'.format(row[0])] = (1, row[3])
            report += u'\n'

            # compute macro averages
            p_macro = np.average(p, weights = None)
            r_macro = np.average(r, weights = None)
            f1_macro = np.average(f1, weights = None)
            metrics['P_{}'.format("macro")] = (1, p_macro)
            metrics['R_{}'.format("macro")] = (1, r_macro)
            metrics['F1_{}'.format("macro")] = (1, f1_macro)
            report += row_fmt.format(last_lines_heading[0],
                                     p_macro,
                                     r_macro,
                                     f1_macro,
                                     total_s,
                                     width=width, digits=digits)

            # compute weighted macro average
            label_weights = map(lambda x : x/(total_s * 1.0), label_weights)
            p_weighted_average = np.average(p, weights = label_weights)
            r_weighted_average = np.average(r, weights = label_weights)
            f1_weighted_average = np.average(f1, weights = label_weights)
            metrics['P_{}'.format("weighted_macro")] = (1, p_weighted_average)
            metrics['R_{}'.format("weighted_macro")] = (1, r_weighted_average)
            metrics['F1_{}'.format("weighted_macro")] = (1, f1_weighted_average)
            report += row_fmt.format(last_lines_heading[1],
                                     p_weighted_average,
                                     r_weighted_average,
                                     f1_weighted_average,
                                     total_s,
                                     width=width, digits=digits)
            # micro average
            metrics['P_{}'.format("micro")] = (1, p_micro)
            metrics['R_{}'.format("micro")] = (1, r_micro)
            metrics['F1_{}'.format("micro")] = (1, f1_micro)
            report += row_fmt.format(last_lines_heading[2],
                                     p_micro,
                                     r_micro,
                                     f1_micro,
                                     total_s,
                                     width=width, digits=digits)

            self.logger.info("P,R,F1 report as follows:\n {}".format(report))
            # only plot it at dev and test time, not during training.
            if self.gen_confusing_matrix:

                self.logger.info("Generate confusing matrix photo.")
                # Compute confusion matrix
                conf_matrix = confusion_matrix(label_true, label_pred)
                np.set_printoptions(precision=2)

                # Plot non-normalized confusion matrix
                plt.figure()
                self.plot_confusion_matrix(conf_matrix, classes=self.brief_classes,
                                      title='Confusion matrix when seq labeling, without normalization')
                wo_norm_fig_path = os.path.join(self.result_dir, '{}_wo_norm.png'.format(self.result_prefix))
                plt.savefig(wo_norm_fig_path)

                # Plot normalized confusion matrix
                plt.figure()
                self.plot_confusion_matrix(conf_matrix, classes=self.brief_classes, normalize=True,
                                      title='Normalized confusion matrix when seq labeling')

                norm_fig_path = os.path.join(self.result_dir, '{}_w_norm.png'.format(self.result_prefix))
                plt.savefig(norm_fig_path)

        else:
            self.logger.warn('invalid total_correct_in_all')

        return metrics

    def eval_metrics_for_multiclass(self, predicted_answers):
        """
        calculate the evaluations
        get t, op 10
        count correct in top10 predictions
        total correct in predictions
        total number predictions are just 10 * #sample
        total correct in all candidates
        Assuming that the 'pred_score' has been assigned already, which can be probability and score.
        """
        total_correct_in_all = 0
        total_pred_in_all = len(predicted_answers)
        # initial a dict for total correct in topK  counting.
        total_correct_in_topK = dict([(i, 0) for i in self.topK_list])
        total_pred_in_topK = dict([(i, 0) for i in self.topK_list])
        max_topK = max(self.topK_list)
        label_pred = []
        label_true = []
        label_weights = []
        digits = 3
        metrics = {}

        for e_id, sample in predicted_answers.iteritems():
            # get all correct ids
            correct_label_indices = sample['correct_labels']
            # current case, we only have a majority lable for the correct label
            label_true.append(correct_label_indices[0])
            # counting all correct for each sample
            total_correct_in_all += len(correct_label_indices)
            # select topK
            sorted_probs_max_topK = sorted(sample['pred_probs'], reverse=True, key=lambda x: x['prob'])[:max_topK]
            top1_pred = sorted_probs_max_topK[0]
            label_pred.append(top1_pred['label_index'])

            # for all topK predictions
            for i in range(len(sorted_probs_max_topK)):
                pred = sorted_probs_max_topK[i]
                for topK in self.topK_list:
                    if i >= topK:
                        continue
                    else:
                        total_pred_in_topK[topK] += 1
                        if pred['label_index'] in correct_label_indices:
                            total_correct_in_topK[topK] += 1

        if total_correct_in_all != 0:
            # recall@K
            recall_at_K = dict([(k, total_correct_in_topK[k] / (total_correct_in_all * 1.0)) for k in self.topK_list])
            # assign recall@K into metrics
            for k, v in recall_at_K.items():
                # Jie
                # 1 means the greater the better.
                # -1 means the smaller the better.
                metrics['R@{}'.format(k)] = (1, v)

            self.logger.info('total_correct_in_all = {}, correct_in_topK = {}, recall@K = {}'.format(total_correct_in_all, sorted(total_correct_in_topK.items()), sorted(recall_at_K.items())))
            # here return all the p,r,f for each label, then we compute the micro average later.
            p, r, f1, s = precision_recall_fscore_support(label_true, label_pred, beta=1.0, labels=range(self.num_classes), average=None)
            total_s = np.sum(s)
            p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(label_true, label_pred, beta=1.0, labels=range(self.num_classes), average='micro')
            last_lines_heading = ['macro / total', 'weighted_mac / total', 'micro / total']
            target_names = self.classes
            name_width = max(len(cn) for cn in target_names)
            width = max(name_width, max([len(x) for x in last_lines_heading]), digits)

            headers = ["precision", "recall", "f1-score", "support"]
            head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
            report = head_fmt.format(u'', *headers, width=width)
            report += u'\n\n'
            row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
            rows = zip(target_names, p, r, f1, s)
            for row in rows:
                label_weights.append(row[4])
                report += row_fmt.format(*row, width=width, digits=digits)
                metrics['P_{}'.format(row[0])] = (1, row[1])
                metrics['R_{}'.format(row[0])] = (1, row[2])
                metrics['F1_{}'.format(row[0])] = (1, row[3])
            report += u'\n'

            # compute macro averages
            p_macro = np.average(p, weights = None)
            r_macro = np.average(r, weights = None)
            f1_macro = np.average(f1, weights = None)
            metrics['P_{}'.format("macro")] = (1, p_macro)
            metrics['R_{}'.format("macro")] = (1, r_macro)
            metrics['F1_{}'.format("macro")] = (1, f1_macro)
            report += row_fmt.format(last_lines_heading[0],
                                     p_macro,
                                     r_macro,
                                     f1_macro,
                                     total_s,
                                     width=width, digits=digits)

            # compute weighted macro average
            label_weights = map(lambda x : x/(total_s * 1.0), label_weights)
            p_weighted_average = np.average(p, weights = label_weights)
            r_weighted_average = np.average(r, weights = label_weights)
            f1_weighted_average = np.average(f1, weights = label_weights)
            metrics['P_{}'.format("weighted_macro")] = (1, p_weighted_average)
            metrics['R_{}'.format("weighted_macro")] = (1, r_weighted_average)
            metrics['F1_{}'.format("weighted_macro")] = (1, f1_weighted_average)
            report += row_fmt.format(last_lines_heading[1],
                                     p_weighted_average,
                                     r_weighted_average,
                                     f1_weighted_average,
                                     total_s,
                                     width=width, digits=digits)
            # micro average
            metrics['P_{}'.format("micro")] = (1, p_micro)
            metrics['R_{}'.format("micro")] = (1, r_micro)
            metrics['F1_{}'.format("micro")] = (1, f1_micro)
            report += row_fmt.format(last_lines_heading[2],
                                     p_micro,
                                     r_micro,
                                     f1_micro,
                                     total_s,
                                     width=width, digits=digits)

            self.logger.info("P,R,F1 report as follows:\n {}".format(report))
            # only plot it at dev and test time, not during training.
            if self.gen_confusing_matrix:

                self.logger.info("Generate confusing matrix photo.")
                # Compute confusion matrix
                conf_matrix = confusion_matrix(label_true, label_pred)
                np.set_printoptions(precision=2)

                # Plot non-normalized confusion matrix
                plt.figure()
                self.plot_confusion_matrix(conf_matrix, classes=self.brief_classes, ori_fmt='d',
                                      title='Confusion matrix, without normalization')
                wo_norm_fig_path = os.path.join(self.result_dir, '{}_wo_norm.png'.format(self.result_prefix))
                plt.savefig(wo_norm_fig_path)

                # Plot normalized confusion matrix
                plt.figure()
                self.plot_confusion_matrix(conf_matrix, classes=self.brief_classes, ori_fmt='d', normalize=True,
                                      title='Normalized confusion matrix')

                norm_fig_path = os.path.join(self.result_dir, '{}_w_norm.png'.format(self.result_prefix))
                plt.savefig(norm_fig_path)

        else:
            self.logger.warn('invalid total_correct_in_all')

        return metrics

    def eval_label_confusion(self, cm):
        if self.gen_confusing_matrix:
            self.logger.info("Generate confusing matrix photo for label emb")
            np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            plt.figure()
            self.plot_confusion_matrix(cm, classes=self.brief_classes, ori_fmt='0.2f', normalize=False, title='label embeding confusion matrix')
            label_emb_confusion_path = os.path.join(self.result_dir, '{}.png'.format(self.result_prefix+'_label_emb'))
            plt.savefig(label_emb_confusion_path)

    def plot_confusion_matrix(self, cm, classes, ori_fmt,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.logger.info("Normalized confusion matrix")
        else:
            self.logger.info('Confusion matrix, without normalization')

        self.logger.info("confusing_matrix is \n {}".format(cm))

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation='vertical')
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else ori_fmt
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    def save_predicts(self, predicted_answers):
        """
            self.result_dir: directory to save predicted answers, answers will not be saved if None
            self.result_prefix: prefix of the file for saving predicted answers,
        """
        # save predicted_answers
        if self.result_dir is not None and self.result_prefix is not None:
            result_file = os.path.join(self.result_dir, self.result_prefix + '.json')
            final_results=[]
            with open(result_file, 'w') as fout:
                for e_id, pred_answer in predicted_answers.iteritems():
                    final_results.append(self.make_pred_output(pred_answer))
                fout.write(json.dumps(final_results, encoding='utf-8', indent=4, ensure_ascii=False) + '\n')
            self.logger.info('Saving {} results to {}'.format(self.result_prefix, result_file))

    def make_pred_output(self, sample):
        """
        Output specified keys for a sample with all the candidate answers, and the predicted topK answer
        This same format will also be loaded as enemsble files for bagging ensemble.
        """
        output_sample = copy.deepcopy(sample)
        # output_sample.pop('profile', None)
        for msg in output_sample['messages-so-far']:
            # msg.pop('tokenized_utterance', None)
            if not isinstance(msg['tokenized_utterance'], basestring):
                msg['tokenized_utterance'] = ' '.join(msg['tokenized_utterance'])
            msg.pop('utterance_token_ids', None)

        for correct_answer in output_sample['options-for-correct-answers']:
            # correct_answer.pop('tokenized_utterance', None)
            if not isinstance(correct_answer['tokenized_utterance'], basestring):
                correct_answer['tokenized_utterance'] = ' '.join(correct_answer['tokenized_utterance'])
            correct_answer.pop('utterance_token_ids', None)

        if 'SEQTAG' in self.decode_goal:
            pass
        else:
            output_sample['pred_probs'] = sorted(output_sample['pred_probs'], key=lambda o: o['prob'], reverse=True)
        return output_sample

    def ensemble_by_ranksum(self, ensemble_predicted_files):
        predicted_samples = {}
        for predict_file in ensemble_predicted_files:
            with open(predict_file,"r") as fin:
                samples = json.load(fin)
                for sample in samples:
                    e_id = sample['example-id']
                    if e_id in predicted_samples:
                        aggregated_sample = predicted_samples[e_id]
                        for ans in sample['options-for-next']:
                            ans_id = ans['candidate-id']
                            for aggregated_ans in sample['options-for-next']:
                                if aggregated_ans['candidate-id'] == ans_id:
                                    aggregated_ans['pred_score'] += ans['pred_score']
                    else:
                        predicted_samples[e_id] = sample
        return predicted_samples

    def ensemble_by_votetop(self, ensemble_predicted_files):
        """
        Given predicted results for ensembling, we always select the max-voted top, after we select it, than just remove it from the rest of candidates.
        """
        predicted_samples = {}
        samples_arrs = []
        file_count = len(ensemble_predicted_files)
        if file_count != 0:
	    # read every predicted files, and sort the answers
            for predict_file in ensemble_predicted_files:
                with open(predict_file, "r") as fin:
                    samples = json.load(fin)
                    samples_dict = {}
                    # sort the answers for every sample
                    for sample in samples:
                        e_id = sample['example-id']
                        # all answers have been sorted according tot he pred_scores
                        sample['options-for-next'] = sorted(sample['options-for-next'], reverse=True, key=lambda x: x['pred_score'])
                        samples_dict[e_id] = sample
                    samples_arrs.append(samples_dict)
            length = len(samples_arrs[0])
            if length != 0:
                # Always vote for top for every sample, remove the top, and vote again
                for e_id in samples_arrs[0].keys():
                    new_sample = copy.deepcopy(samples_arrs[0][e_id])
                    # for every sample, just select and remove the top 1, iteratively
                    ranked_anss_from_each_ensemble = list(map(lambda samples_dict: samples_dict[e_id]['options-for-next'], samples_arrs))
                    total_ans = len(new_sample['options-for-next'])
                    # empty the options-for-next
                    new_sample['options-for-next'] = []
                    for order in range(total_ans):
                        top = self.select_top_and_remove(ranked_anss_from_each_ensemble)
                        new_sample['options-for-next'].append(top)
                    predicted_samples[e_id] = new_sample
            else:
                # only one for ensembling, then just copy them
                for sample in sample_arrs[0]:
                    e_id = sample['example-id']
                    predicted_samples[e_id] = sample

        return predicted_samples

    def select_top_and_remove(self, ranked_anss_from_each_ensemble):
        """
        Given the N ranked_list of ans from each ensemble model, we select the top between every top
        in the rest of ranked_anss_from_each_ensemble.
        """
        top_ans_for_each_ensemble_sample = list(map(lambda anss: anss[0], ranked_anss_from_each_ensemble))
        top = max(top_ans_for_each_ensemble_sample, key=lambda x: x['pred_score'])
        # add top into the new_sample ans
        cid = top['candidate-id']

        def remove_ans(anss, cid):
            for ans in anss:
                if cid == ans['candidate-id']:
                    anss.remove(ans)
                    break
            return anss
        # remove the top answer from the ranked ans list, and select the top again.
        ranked_anss_from_each_ensemble = map(lambda anss: remove_ans(anss, cid), ranked_anss_from_each_ensemble)

        return top
