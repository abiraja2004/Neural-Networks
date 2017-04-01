#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 13:19:00 2017
@author: chosenone

"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
import data_utils

from tensorflow.contrib import rnn

# parameters

learning_rate = 0.001
training_iter = 20
display_interval = 1
batch_size = 64

# neural-network parameters

nn_input = 200 
word_level_hidden_size = 200
nn_classes = 1
sentence_dim = 200
document_dim = 200


# helper function

def weight_variables(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)


with tf.name_scope("input"):
    x = tf.placeholder(tf.float32,shape=[None,None,nn_input],"x-input")
    y = tf.placeholder(tf.int16,shape=[None,1],"y-input")

with tf.name_scope("variables"):
    W_d = weight_variables(shape=[2*sentence_dim,1])
    d_bias = bias_variables(None)
    W_c = weight_variables(shape=[2*sentence_dim,1]) # content
    W_s = weight_variables(shape=[2*sentence_dim],1) # salience
    W_r = weight_variables(shape=[2*sentence_dim],1) # novelty
    
                          
    bias = bias_variables(None)


def HierarchicalBiRNN(x,W_d,d_bias,W_c,W_s,W_r,bias):
    
    for document in x:
        
        sentencens_rep = []
        
        for sentence in document['sentences']:
            
            sentence = tf.reshape(sentence,[len(sentence),-1,nn_input])
            # define gru cell of word level
            gru_fw_cell_word_level = rnn.GRUCell(word_level_hidden_size)
            gru_bw_cell_word_level = rnn.GRUCell(word_level_hidden_size)
            
            outputs_word_level,_fw,_bw = rnn.static_bidirectional_rnn(gru_fw_cell_word_level
                                          ,gru_bw_cell_word_level,sentence,
                                           dtype=tf.float32)
            
            input_sentence_level = tf.reduce_mean(tf.reshape(outputs_word_level,[-1,2*word_level_hidden_size]))
            
            sentencens_rep.append(tf.input_sentence_level)
            
        gru_fw_cell_sentence_level = rnn.GRUCell(sentence_dim)
        gru_bw_cell_sentence_level = rnn.GRUCell(sentence_dim)
        
        sentencens_rep = tf.reshape(sentencens_rep,[len(sentencens_rep),-1,2*word_level_hidden_size])
        outputs_sentences_level,_,_ = rnn.static_bidirectional_rnn(gru_fw_cell_sentence_level,
                                                               gru_fw_cell_sentence_level,
                                                               sentencens_rep,
                                                               dtype=tf.float32)
        outputs_sentences_level = tf.reshape(outputs_sentences_level,[-1,2*sentence_dim])
        
        d = tf.nn.tanh(tf.matmul(tf.reduce_mean(outputs_sentences_level),W_d)) + d_bias        
        
    pass






















