#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
A Bidirectional Recurrent Neural Network(LSTM) implementation using Tensorflow
by Lebron.Ran

Created on Thu Mar 16 20:58:18 2017
@author: lebron.ran
"""

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

#import dataset

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

# initial hyper*** parameters

learning_rate = 0.001
train_iters = 1000
batch_size = 128
display_interval = 10

# neural neywork' parameters

nn_input = 28
nn_time_step = 28
nn_hidden_size = 128
nn_classes = 10

# helper function

def weight_variables(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

# tf Graph

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32,shape=[None,nn_time_step,nn_input],name="x-input")
    y = tf.placeholder(tf.float32,shape=[None,nn_classes],name="y-input")

with tf.name_scope("variables"):
    weights = weight_variables([2*nn_hidden_size,nn_classes])
    bias = bias_variables([nn_classes])

def BiRNN(x,weights,bias):
    
    # prepare the data for bi_rnn function
    
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,nn_input])
    x = tf.split(x,nn_time_step,0)
    
    # define lstm cell with tensorflow
    
    gru_fw_cell = rnn.GRUCell(nn_hidden_size)
    gru_bw_cell = rnn.GRUCell(nn_hidden_size)
    
#==============================================================================
#     # forward lstm cell
#     lstm_fw_cell = rnn.BasicLSTMCell(nn_hidden_size,forget_bias=1.0)
#     # backward lstm cell
#     lstm_bw_cell = rnn.BasicLSTMCell(nn_hidden_size,forget_bias=1.0)
#==============================================================================
    
    outputs,_fw,_bw = rnn.static_bidirectional_rnn(gru_fw_cell,gru_bw_cell,x,
                                           dtype=tf.float32)
    
    return tf.matmul(outputs[-1],weights) + bias

with tf.name_scope("prediction"):
    y_hat = BiRNN(x,weights,bias)
    
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat,
                                                                  labels=y))
    
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    
with tf.name_scope("evaluation"):
    correct_preds = tf.equal(tf.argmax(y_hat,axis=1),tf.argmax(y,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds,dtype=tf.float32))


initializers = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializers)
    step = 0
    while step < train_iters:
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        
        batch_x = batch_x.reshape((batch_size,nn_time_step,nn_input))
        
        loss,acc,_ = sess.run([cost,accuracy,optimizer],feed_dict={x:batch_x,y:batch_y})
        
        if step % display_interval == 0:
#==============================================================================
#             loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y})
#==============================================================================
            print('Iter: %d,Mini-Batch Loss: %.3f,Accuracy: %.3f' % (step,loss,acc))
        
        step += 1
            

    print("Evaluating on Test Set....")
            
    test_len = 200
    
    test_x = mnist.test.images[:test_len].reshape((-1,nn_time_step,nn_input))
    test_y = mnist.test.labels[:test_len]
    
    
    print("Accuracy on Test Set: %.3f" % (sess.run(accuracy,feed_dict={x:test_x,
                                                  y:test_y})))        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        










    
    
    
        

















































