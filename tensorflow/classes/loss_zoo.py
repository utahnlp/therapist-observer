# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 15:06:56 jcao>
# --------------------------------------------------------------------
# File Name          : loss_zoo.py
# Original Author    : jiessie.cao@gmail.com
# Description        : All kinds of loss function regression and rank loss
# --------------------------------------------------------------------

import re
import tensorflow as tf

def _compute_pairwise_hinge_loss(loss_func_name, logits, correct_labels, num_classes, margin_prob=0.01, pw_alpha=1.0, focal_loss_gama=0):
    """
    # Jie
    The hingle loss for pairwise max-marging
    margin_structural_loss - KL(Pc, P')
    p_c - p_i > margin - \epsilon_i, \epsilon >= 0
    max(0, margin+p_i-p_c), for all i != c
    # KL(Pc, P') is the difference between current and target prob distribution
    # Cross entropy H(Pc, P') - H(Pc) = KL(Pc, P')
    # In this multiclass case, only 1.0 for the correct one, others are 0.0, H(Pc) = 0
    # We also can consider KL(Pc, P') as the difference of the score of different distribution
    # marginal_structual loss is the structure loss, then it is just the structured hinge loss.
    # min(sum(margin_structural + KL(Pc, P')))
    # min(sum(margin_structural + Xentropy(Pc, P')))
    """
    # [batch_size, max_candidate_answer]
    pred_probs = tf.nn.softmax(logits)
    correct_index_with_batch_index = tf.concat([tf.expand_dims(tf.range(tf.shape(pred_probs)[0]), -1), tf.expand_dims(correct_labels, -1)], -1)
    # [batch_size]
    correct_probs = tf.gather_nd(pred_probs, correct_index_with_batch_index)
    # [batch_size, 1]
    correct_probs_expand = tf.expand_dims(correct_probs, -1)
    # [batch_size, num_classes]
    dup_correct_probs = tf.ones_like(pred_probs, tf.float32) * correct_probs_expand
    # a single float by reducemean the whole matrix
    # relu is max(0, x), 0 < x <= 1.1 for prob
    # if less than margin, loss is 0, otherwise is 1, don't care about the prob
    # hingloss is [1], to let the probability meet the binary goal
    # for every sample, we also counted the correct label itself, hence, we minus 1 for each sample.
    if re.match("[^@]+@\d+", loss_func_name):
        K = int(loss_func.split('@')[1])
    else:
        K = 0

    margin_unsatisfied_pair_counts = tf.reduce_sum(tf.sign(tf.nn.relu(pred_probs + margin_prob - dup_correct_probs)), -1) - 1
    # only count the count of exceeded paris as loss
    unsatisfied_gap_counts_by_K = margin_unsatisfied_pair_counts - K
    pw_hinge_loss = tf.reduce_mean(unsatisfied_gap_counts_by_K)
    # xentropy meet the regression goal
    # default pw_alpha=1.0, focal_loss_gama=0.0
    # when pw_loss is small, the xentropy is also small, otherwise, when pw_loss is larger, even if the prob is good, we still make it punished.
    xentropy_loss = tf.reduce_mean(tf.pow(unsatisfied_gap_counts_by_K/(num_classes + 0.0), focal_loss_gama) * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_labels, logits=logits))
    loss = pw_alpha * pw_hinge_loss + xentropy_loss
    return loss

def _compute_pwscore_hinge_loss(loss_func_name, logits, correct_labels, num_classes, margin_score=1.0, pw_alpha=1.0, focal_loss_gama=0):
    """
    # Jie
    The hingle loss for pairwise max-marging
    margin_structural_loss - KL(Pc, P')
    p_c - p_i > margin - \epsilon_i, \epsilon >= 0
    max(0, margin+p_i-p_c), for all i != c
    # KL(Pc, P') is the difference between current and target prob distribution
    # Cross entropy H(Pc, P') - H(Pc) = KL(Pc, P')
    # In this multiclass case, only 1.0 for the correct one, others are 0.0, H(Pc) = 0
    # We also can consider KL(Pc, P') as the difference of the score of different distribution
    # marginal_structual loss is the structure loss, then it is just the structured hinge loss.
    # min(sum(margin_structural + KL(Pc, P')))
    # min(sum(margin_structural + Xentropy(Pc, P')))
    """
    # [batch_size, max_candidate_answer]
    correct_index_with_batch_index = tf.concat([tf.expand_dims(tf.range(tf.shape(logits)[0]), -1), tf.expand_dims(correct_labels, -1)], -1)
    # [batch_size]
    correct_scores = tf.gather_nd(logits, correct_index_with_batch_index)
    # [batch_size, 1]
    correct_scores_expand = tf.expand_dims(correct_scores, -1)
    # [batch_size, num_classes]
    dup_correct_scores = tf.ones_like(logits, tf.float32) * correct_scores_expand

    # a single float by reducemean the whole matrix
    # relu is max(0, x), 0 < x <= 1.1 for prob
    # just use the score difference as hinge loss
    # for every sample, we add an unnessary loss for the correct label.
    unsatified_pair_gap_scores = tf.reduce_sum(tf.nn.relu(logits + margin_score - dup_correct_scores), -1)
    pw_hinge_loss = tf.reduce_mean(unsatified_pair_gap_scores - margin_score)
    # xentropy meet the regression goal
    xentropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_labels, logits=logits))
    loss = pw_alpha * pw_hinge_loss + xentropy_loss
    return loss

def _compute_xentropy_with_logits(logits, correct_labels):
    """
    # currently, we use the standard xentropy loss, without focal_loss_gama
    """
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_labels, logits=logits))

def _compute_weighted_xentropy_with_logits(logits, correct_labels, weights):
    """
    we multiple weights for the loss of each example.
    weights are the mutlitply factor for each different labels, usually are normalized inverse frequency. Shape is [num_classes]
    First, tile it into [num_classes, 1]
    Second, expand labels [batch_size] into [batch_size, number_classes]
    Then matmul expand_labels with weights, we can get a [batch_size, 1]
    Then sequenze[batch_size, 1] into [batch_size], which then can be mutplied with loss
    """
    num_classes = tf.shape(weights)[0]
    expanded_weights = tf.expand_dims(weights,-1)
    expanded_labels = tf.one_hot(correct_labels, num_classes)
    weights_batch = tf.matmul(expanded_labels, expanded_weights, transpose_b = False)
    squeeze_batch_weights = tf.squeeze(weights_batch)
    return tf.reduce_mean(squeeze_batch_weights * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_labels, logits=logits))

def _compute_weighted_focal_loss(logits, probs, correct_labels, weights, gama):
    """
    https://arxiv.org/abs/1708.02002
    probs, [batach_size, num_class], the probabilities for all the classs, and all the example in the batch
    weights, [num_class] are the \alpha balance, just as the usuall weighted \alpha balnance, one per class
    gama , [num_class] is power factor for hard-easy examples. [1-5], usually, we use the same gama for every class.
    When gama is zero, then it downgrade to weighted xentropy loss
    """
    num_classes = tf.shape(weights)[0]
    expanded_weights = tf.expand_dims(weights,-1)
    # [batch_size, num_classes]
    expanded_labels = tf.one_hot(correct_labels, num_classes)
    weights_batch = tf.matmul(expanded_labels, expanded_weights, transpose_b = False)
    squeeze_batch_weights = tf.squeeze(weights_batch)
    expanded_gama = tf.expand_dims(gama, -1)
    # [batch_size]
    squeeze_gama_batch = tf.squeeze(tf.matmul(expanded_labels, expanded_gama, transpose_b = False))
    # [batch_size]
    squeeze_correct_probs = tf.squeeze(tf.matmul(tf.expand_dims(probs, 1), tf.expand_dims(expanded_labels, -1), transpose_b = False))
    # [batch_size]
    squeeze_gap_probs = 1 - squeeze_correct_probs
    # [batch_size]
    squeeze_gama_factor = tf.pow(squeeze_gap_probs, squeeze_gama_batch)
    return tf.reduce_mean(squeeze_batch_weights * squeeze_gama_factor * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_labels, logits=logits))


def focal_cross_entropy_loss_probs(prediction, labels, pos_w=0.995, gama=2, scope=None):
    """
    https://arxiv.org/abs/1708.02002
    focal loss
    """
    with tf.name_scope(scope, "focal_xentropy_loss"):
        all_ones = tf.fill(tf.shape(prediction), 1.0)
        pos_weight = tf.get_variable(
            "W_loss", trainable=False,
            initializer=tf.constant(pos_w)
        )
        # cross_entropy only penalty for the correct class,.
        weight_for_loss = [1 - pos_weight, pos_weight]
        losses = -tf.reduce_mean(weight_for_loss * labels * tf.pow((all_ones - prediction), gama) * tf.log(prediction))
        return losses
