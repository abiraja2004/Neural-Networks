#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:34:35 2017

@author: chosenone
"""

import tensorflow as tf
import numpy as np

inputs =  [[[1, 1, 1], [2, 2, 2]],
           [[3, 3, 3], [4, 4, 4]],
           [[5, 5, 5], [6, 6, 6]],
           [[7, 7, 7], [8, 8, 8]]]


# the express below only can be used with ndarrays


#==============================================================================
# print inputs[:,1,:]
#==============================================================================

sess = tf.Session()

i = tf.train.range_input_producer(5).dequeue()

print sess.run(tf.strided_slice(inputs,[0,0,0],[3,2,2],[1,1,2]))

#note of tf.concat : all you need to do is sum over the given axis while keeping other dim unchanged 

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]

print tf.concat([t1, t2], 0).shape #==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
print tf.concat([t1, t2], 1).shape #==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]              
print tf.concat(inputs,axis=1).shape





