#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:42:52 2017
by Lebron.Ran

CNN for text classification

Structure:
A embedding lay,followed by a convolutional lay and a max-pooling lay,at last 
softmax layer.
    
"""

import tensorflow as tf
import numpy as np


class CNN4Text(object):
    
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size
                 ,filter_sizes,num_filters,l2_reg_lambda=0.0):
        # placeholders for input,output and dropout
        
        with tf.name_scope(name="input"):
            self._input_x = tf.placeholder(tf.int32,[None,sequence_length],name='x')
            self._input_y = tf.placeholder(tf.float32,[None,num_classes],name = 'y')
            self._keep_prob = tf.placeholder(tf.float32,name = 'keep_prop')
        
        # l2 regularization if needed
        
        l2_loss  = tf.constant(0.0,tf.float32)
        
        # embedding layer
        
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            self._embeddings = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),
                                           name="embedding")
            self._embedded_words = tf.nn.embedding_lookup(self._embeddings,self._input_x)
            self._embedded_words_expanded  = tf.expand_dims(self._embedded_words,-1)
            
        # creat a convolution layer and pool layer for each filter size
        
        pooled_output = []
        
        for i,filter_size in enumerate(filter_sizes):
            
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                
                # convolution layer
                filter_shape = [filter_size,embedding_size,1,num_filters]
                W  = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
                b = tf.Variable(tf.constant(0.1,shape=[num_filters]))
                
                # convolution operation
                conv = tf.nn.conv2d(self._embedded_words_expanded,
                                    W,
                                    strides=[1,1,1,1],
                                    padding="VALID",
                                    name="conv")
                
                # apply activation function
                
                h = tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                
                # max-pooling layers
                
                pooled = tf.nn.max_pool(h,
                                        ksize=[1,sequence_length - filter_size + 1,1,1],
                                        strides=[1,1,1,1],
                                        padding="VALID",
                                        name="pool")
                
                pooled_output.append(pooled)
                
        # combine all the pooled output
        
        num_filters_total = num_filters * len(filter_sizes)
        
        self.h_pool = tf.concat(pooled_output,axis=3)
        
        self.h_pool_flatened = tf.reshape(self.h_pool,[-1,num_filters_total])
        
        
        # add dropout 
        
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flatened,self._keep_prob)
            
        # final scores and predictions
        
        with tf.name_scope("output"):
            
            W = tf.get_variable(name="W",shape=[num_filters_total,num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1,tf.float32,shape=[num_classes],name="b"))
            
            l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b) # really?
            
            self._scores = tf.nn.xw_plus_b(self.h_pool_flatened,W,b,name="scores")
            self._predictions = tf.argmax(self._scores,axis=1,name="predictions")                           
        
        # calculate cross-entropy
                             
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self._scores,labels=self._input_y)
            self._loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        
        # Accuracy

        correct_predictions = tf.equal(self._predictions,tf.argmax(self._input_y,axis=1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy")                                
            
                              
                                      
        
        
        
        
        
        
        
        
        
                
                
                
                
                
                
                
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    

