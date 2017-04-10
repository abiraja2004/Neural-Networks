#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:35:49 2017
@author: chosenone

A model with hierarchical bidirectional GRU for text summarization.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import time

from tensorflow.contrib import rnn

def BidirectionalGRU4Summ(object):
    
    def __init__(self,document_len,sequence_length,vocab_size,embedding_size,
                 word_hidden_size,sentence_hidden_size,l2_reg_lambda=0.0):
        
        # place holder for input and output and dropout
        with tf.name_scope("input"):
            self._input_x = tf.placeholder(tf.int32,[document_len,sequence_length],
                                           name="input_x")
            self._input_y = tf.placeholder(tf.float32,[document_len,1],name="input_y")
            self._keep_prob = tf.placeholder(tf.float32,name="keep_prob")
        
        # l2-regularization if needed
        l2_loss = tf.constant(0.0,tf.float32,name="l2_loss")
        
        # embedding layer
        
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            self._embeddings = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),
                                           name="embedding")
            self._embedded_words = tf.nn.embedding_lookup(self._embeddings,self._input_x)
        
        with tf.name_scope("bi-gru"):
            
            # re-format the data
            x = tf.transpose(self._input_x,[1,0,2])
            x = tf.reshape(x,[-1,embedding_size])
            x = tf.split(x,sequence_length,0)
            
            # define the GRU cell in tensorflow
            gru_fw_cell = rnn.GRUCell(word_hidden_size,activation=tf.nn.relu)
            gru_bw_cell = rnn.GRUCell(word_hidden_size,activation=tf.nn.relu)
            
            outputs,_fw,_bw = rnn.static_bidirectional_rnn(gru_fw_cell,gru_bw_cell,x,dtype=tf.float32)
        
            
        
        
        
        
        
        
        