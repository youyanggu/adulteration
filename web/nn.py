#!/usr/bin/env python
"""
Author: Youyang Gu (yygu@mit.edu)

Neural Network framework built on top of Theano. This is a skeleton version that 
takes as input the pre-trained weights and computes the output given the input.
It has a hidden layer (where m is the number of hidden layers) and a output layer.

"""

import numpy as np
import theano
import theano.tensor as T

class OutputLayer(object):
    def __init__(self, inp, n_in, n_out, W_values, b_values):
        self.inp = inp
        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(inp, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

class HiddenLayer(object):
    def __init__(self, inp, n_in, n_out, W_values, b_values, activation=None):
        self.inp = inp
        W = theano.shared(value=W_values, name='W', borrow=True)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(inp, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class NN(object):
    def __init__(self, inp, n_in, m, n_out, W_hid=None, b_hid=None, W_out=None, b_out=None):
        self.inp = inp
        self.hiddenLayer = HiddenLayer(
            inp=self.inp,
            n_in=n_in,
            n_out=m,
            W_values=W_hid,
            b_values=b_hid,
            activation=T.tanh
        )
        self.outputLayer = OutputLayer(
            inp=self.hiddenLayer.output,
            n_in=m,
            n_out=n_out,
            W_values=W_out,
            b_values=b_out
        )

