#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:24:45 2017

@author: chosenone
"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


x = tf.placeholder(tf.float32,shape=[None,784])

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y_hat = tf.nn.softmax(tf.matmul(x,W) + b)

y = tf.placeholder(tf.float32,shape=[None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat),reduction_indices=[1]))

correct_predictions = tf.equal(tf.arg_max(y_hat,dimension=1),tf.arg_max(y,dimension=1))

accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))

#===============================more stable====================================
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits()
#==============================================================================


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

#train our model
#Batch Gradient Descent 

for i in range(10000):
    batch_x_train,batch_y_train = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_x_train,y:batch_y_train})   
    
    

# evaluate the correct accuracy

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

    







