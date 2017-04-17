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

class BidirectionalGRU4Summ(object):
    
    def __init__(self,document_len,sequence_length,vocab_size,embedding_size,
                 word_hidden_size,sentence_hidden_size,doc_rep_size,
                 word_embedding=None,fine_tune=False,l2_reg_lambda=0.0):
        
        if fine_tune and word_embedding == None:
            raise("Value Error:there must be a copy of initial value of word embedding")
        
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
            # fine-tuning or not
            if not fine_tune:
                self._embeddings = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),
                                               name="embedding")
            else:
                self._embeddings = tf.Variable(word_embedding,name="embedding-fine-tuned")
                
            self._embedded_words = tf.nn.embedding_lookup(self._embeddings,self._input_x)    
        
        with tf.name_scope("bi-gru-word"),tf.variable_scope("bi-gru-word"):
            
            # re-format the data
            x = tf.transpose(self._embedded_words,[1,0,2])
            x = tf.reshape(x,[-1,embedding_size])
            x = tf.split(x,sequence_length,0)
            
            # define the GRU cell in tensorflow
            gru_fw_cell = rnn.GRUCell(word_hidden_size,activation=tf.nn.relu)
            gru_bw_cell = rnn.GRUCell(word_hidden_size,activation=tf.nn.relu)
            
            outputs,_fw,_bw = rnn.static_bidirectional_rnn(gru_fw_cell,gru_bw_cell,x,dtype=tf.float32)
            
            # shape of outputs:[time][batch][cell_fw.output_size + cell_bw.output_size]
            output_trans = tf.transpose(outputs,[1,0,2])
            input_to_sentence_level = tf.reduce_mean(output_trans,axis=1) # shape:[batch][size*2] 
            
        with tf.name_scope("bi-gru-sentence"),tf.variable_scope("bi-gru-sentence"):
            
            input_to_sentence_level = tf.reshape(input_to_sentence_level,[-1,2*word_hidden_size])
            input_to_sentence = tf.split(input_to_sentence_level,document_len,0)
            gru_fw_cell_sentence = rnn.GRUCell(sentence_hidden_size,activation=tf.nn.relu)    
            gru_bw_cell_sentence = rnn.GRUCell(sentence_hidden_size,activation=tf.nn.relu)
            
            outputs,_,_ = rnn.static_bidirectional_rnn(gru_fw_cell_sentence,gru_bw_cell_sentence,
                                                       input_to_sentence,dtype=tf.float32)
        
            outputs = tf.reshape(outputs,[-1,2*sentence_hidden_size])
        
        with tf.name_scope("document-representation"):
            average_sent_rep = tf.reduce_mean(outputs,axis=0)
            average_sent_rep = tf.reshape(average_sent_rep,[-1,2*sentence_hidden_size])
            W_d = tf.get_variable("W_d",[2*sentence_hidden_size,doc_rep_size],dtype=tf.float32)
            b_d = tf.get_variable("b_d",[doc_rep_size],dtype=tf.float32)
            doc_rep = tf.nn.tanh(tf.matmul(average_sent_rep,W_d) + b_d)
            doc_rep = tf.reshape(doc_rep,[-1,1])
            
        with tf.name_scope("prediction"):
            W_c = tf.get_variable("W_c",[2*sentence_hidden_size,1],dtype=tf.float32)
            W_s = tf.get_variable("W_s",[2*sentence_hidden_size,doc_rep_size],dtype=tf.float32)
            W_r = tf.get_variable("W_r",[2*sentence_hidden_size,2*sentence_hidden_size],dtype=tf.float32)
            b_p = tf.get_variable("b_p",[],dtype=tf.float32)
            
            # contents
            contents = tf.matmul(outputs,W_c,name="content")
            # salience             
            salience = tf.matmul(tf.matmul(outputs,W_s),doc_rep,name="salience")
            # redundancy
            redundancy = tf.reduce_mean(tf.matmul(tf.matmul(outputs,W_r)
                         ,tf.transpose(outputs,[1,0])),name="redundancy",axis=1)
            
            scores = tf.nn.sigmoid(contents + salience - redundancy + b_p,name="scores")
            
            self._scores = scores
        with tf.name_scope("loss"):
            
            losses = 1.0 / 2 * tf.reduce_mean(-(self._input_y * tf.log(tf.clip_by_value(self._scores,1e-10,1.0)) 
                                              + (1 - self._input_y) * tf.log(tf.clip_by_value(1 - self._scores,1e-10,1.0))))
            self._loss = losses + l2_reg_lambda * l2_loss
        
            
        
        
        
        
        
        
        
        
        
        