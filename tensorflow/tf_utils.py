# -*- coding:utf8 -*-
# Time-stamp: <2019-06-04 14:47:45 jcao>
# --------------------------------------------------------------------
# File Name          : tf_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : This module implements some extra tools for tensorflow
# --------------------------------------------------------------------

import tensorflow as tf

def _tf_dup_1(source, times):
    """
    used to duplicate the first dimention for N times, usualy to align it with some other tensor.
    e.g. batch_size = 3, while for every batch, we have N different sub examples.
    To predict, we usually hope that those sub examples can be aligned.
    e.g.
    A = [1, 2, 3],
    B = [
          1.1, 1.2, 1.3, 1.4,
          2.1, 2.2, 2.3, 2.4,
          3.1, 3.2, 3.3, 3.4
        ]
    just repeat source = [1,2,3] for 4 times to align with B, then we have
    A_dup = [
              1, 1, 1, 1,
              2, 2, 2, 2,
              3, 3, 3, 3
            ],
    source  : A tensor, 1-D
    times : An interger, a number of times
    """
    return tf.reshape(tf.transpose(tf.reshape(tf.tile(source, [times]), [-1, tf.shape(source)[0]])), [-1])


def _tf_dup_2(source, times, last_dimension):
    """
    just repeat the second dimension of source for 4 times, from
    [
      [1,2],[3,4],[5,6]
    ]
    to
    [
      [1,2],
      [1,2],
      [1,2],
      [1,2],
      [3,4],
      [3,4],
      [3,4],
      [3,4],
      [5,6],
      [5,6],
      [5,6],
      [5,6]
    ]
    source  : A tensor, 3-D
    times : An interger, a number of times
    last_dimension : the size of last dimension, we cannot use tf.shape(source)[-1] for the ability to inference
    """
    return tf.reshape(tf.tile(source, [1, times]), [-1, last_dimension])


def _tf_dup_3(source, times, last_dimension):
    """
    just repeat the second dimension of source for 4 times, from
    [
      [[1,2],[3,4][5,6]],
      [[0,1],[2,3],[4,5]]
    ]
    to
    [
       [[1,2],[3,4],[5,6]],
       [[1,2],[3,4],[5,6]],
       [[1,2],[3,4],[5,6]],
       [[1,2],[3,4],[5,6]],
       [[0,1],[2,3],[4,5]],
       [[0,1],[2,3],[4,5]],
       [[0,1],[2,3],[4,5]],
       [[0,1],[2,3],[4,5]]
    ]
    source  : A tensor, 3-D
    times : An interger, a number of times
    last_dimension : the size of last dimension, we cannot use tf.shape(source)[-1] for the ability to inference
    """
    return tf.reshape(tf.tile(source, [1, times, 1]), [-1, tf.shape(source)[1], last_dimension])
