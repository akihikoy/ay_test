#!/usr/bin/python
#\file    loss_for_error.py
#\brief   Loss function for error model.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.18, 2015
#
#  This implementation is based on:
#      chainer/functions/mean_squared_error.py
#      chainer/functions/relu.py
#
import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class LossForError1(function.Function):

    """Loss function for error model (ver.1)."""

    def __init__(self, beta=0.1):
        self.beta = numpy.float32(beta)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        zero = utils.force_type(x0.dtype, 0)
        diff = x1 - x0
        pos_diff = numpy.maximum(zero, diff.ravel())
        #n_pos = max(1,numpy.count_nonzero(pos_diff))
        x0s = x0.ravel()
        loss = (pos_diff.dot(pos_diff) + self.beta*x0s.dot(x0s)) / diff.size
        return numpy.array(loss, numpy.float32),

    def forward_gpu(self, inputs):
        #x0, x1 = inputs
        #ret = cuda.reduce(
            #'const float* x0, const float* x1',
            #'(x0[i] - x1[i]) * (x0[i] - x1[i])',
            #'a+b', '0', 'mse_fwd', numpy.float32)(x0, x1)
        #ret /= x0.size
        #return ret,
        raise Exception("TO IMPLEMENT")

    def backward_cpu(self, inputs, gy):
        x0, x1 = inputs
        zero = utils.force_type(x0.dtype, 0)
        diff = x1 - x0
        pos_diff = numpy.maximum(zero, diff)
        #n_pos = max(1,numpy.count_nonzero(pos_diff))
        coeff = 2. * gy[0] / diff.size
        gx1 = coeff * pos_diff
        gx0 = -gx1 + self.beta * coeff * x0
        return gx0, gx1

    def backward_gpu(self, inputs, gy):
        #x0, x1 = inputs
        #gx0 = cuda.empty_like(x0)
        #gx1 = cuda.empty_like(x1)
        #coeff = gy[0] * (2. / x0.size)
        #cuda.elementwise(
            #'''float* gx0, float* gx1, const float* x0, const float* x1,
               #const float* coeff''',
            #'''gx0[i] = *coeff * (x0[i] - x1[i]);
               #gx1[i] = -gx0[i];''',
            #'mse_bwd')(gx0, gx1, x0, x1, coeff)
        #return gx0, gx1
        raise Exception("TO IMPLEMENT")


def loss_for_error1(x0, x1, beta=0.1):
    """Loss function for error model (ver.1).

    This function computes loss-for-error-model between two variables (x0,x1).
    Note: x0 must be a result of model, x1 must be a data of errors.
    The error is not scaled by 1/2.

    """
    return LossForError1(beta)(x0, x1)








class LossForError2(function.Function):

    """Loss function for error model (ver.2)."""

    def __init__(self, beta=0.1):
        self.beta = numpy.float32(beta)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        zero = utils.force_type(x0.dtype, 0)
        diff = x1 - x0
        pos_diff = numpy.maximum(zero, diff.ravel())
        neg_diff = numpy.minimum(zero, diff.ravel())
        loss = (pos_diff.dot(pos_diff) + self.beta*neg_diff.dot(neg_diff)) / diff.size
        return numpy.array(loss, numpy.float32),

    def forward_gpu(self, inputs):
        #x0, x1 = inputs
        #ret = cuda.reduce(
            #'const float* x0, const float* x1',
            #'(x0[i] - x1[i]) * (x0[i] - x1[i])',
            #'a+b', '0', 'mse_fwd', numpy.float32)(x0, x1)
        #ret /= x0.size
        #return ret,
        raise Exception("TO IMPLEMENT")

    def backward_cpu(self, inputs, gy):
        x0, x1 = inputs
        zero = utils.force_type(x0.dtype, 0)
        diff = x1 - x0
        pos_diff = numpy.maximum(zero, diff)
        neg_diff = numpy.minimum(zero, diff)
        coeff = 2. * gy[0] / diff.size
        gx1 = coeff * (pos_diff + self.beta*neg_diff)
        gx0 = -gx1
        return gx0, gx1

    def backward_gpu(self, inputs, gy):
        #x0, x1 = inputs
        #gx0 = cuda.empty_like(x0)
        #gx1 = cuda.empty_like(x1)
        #coeff = gy[0] * (2. / x0.size)
        #cuda.elementwise(
            #'''float* gx0, float* gx1, const float* x0, const float* x1,
               #const float* coeff''',
            #'''gx0[i] = *coeff * (x0[i] - x1[i]);
               #gx1[i] = -gx0[i];''',
            #'mse_bwd')(gx0, gx1, x0, x1, coeff)
        #return gx0, gx1
        raise Exception("TO IMPLEMENT")


def loss_for_error2(x0, x1, beta=0.1):
    """Loss function for error model (ver.1).

    This function computes loss-for-error-model between two variables (x0,x1).
    Note: x0 must be a result of model, x1 must be a data of errors.
    The error is not scaled by 1/2.

    """
    return LossForError2(beta)(x0, x1)





