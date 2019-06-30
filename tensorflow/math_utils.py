# -*- coding: utf-8 -*
# Time-stamp: <2019-06-04 14:50:45 jcao>
# --------------------------------------------------------------------
# File Name          : math_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : Utils for math operaters
# --------------------------------------------------------------------

import numpy
numpy.seterr(all='ignore')


def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def dsigmoid(x):
    return x * (1. - x)


def tanh(x):
    return numpy.tanh(x)


def dtanh(x):
    return 1. - x * x


def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    return e / numpy.sum(e, axis=0)


def norm_prob(x):
    return x / numpy.sum(x, axis=0)


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1. * (x > 0)


def acc_sum_len(y, arr):
    for i in range(len(arr)):
        y = y - arr[i]
        if y <= 0.0:
            return i + 1
    return len(arr)
