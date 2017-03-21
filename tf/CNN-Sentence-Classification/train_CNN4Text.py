#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:53:19 2017
@author: chosenone

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_util

from CNN4Text import CNN4Text
from tensorflow.contrib import learn


#==============================================================================
# parameters
#==============================================================================

#==============================================================================
# data parameters
#==============================================================================

tf.flags.DEFINE_float("validation_set_percentage",0.1,
        "the percentage of training examples that will be used for validation set")

tf.flags.DEFINE_string("data_postive_path","./data/rt-polaritydata/rt-polarity.pos",
                       "file path for postive data")

tf.flags.DEFINE_string("data_negative_path","./data/rt-polaritydata/rt-polarity.neg",
                       "file path for negative data")

#==============================================================================
# model hyperparameters
#==============================================================================

tf.flags.DEFINE_integer("embedding_size",128,"the size of word embeeding (default 128)")

tf.flags.DEFINE_integer("num_filters",128,"the number of filters for each filter size(default 128)")

tf.flags.DEFINE_string("filter_sizes","3,4,5","comma-separated filter sizes(default 3,4,5)")

tf.flags.DEFINE_float("keep_prob",0.5,"the probability used for dropout(default 0.5)")

tf.flags.DEFINE_float("l2_reg_lambda",0.0,"the l2 regularization lambda(default 0)")


#==============================================================================
# train parameters
#==============================================================================

tf.flags.DEFINE_integer("batch_size",64,"Batch size (default size 64)")

tf.flags.DEFINE_integer("num_epochs",200,"Epoch sizes(default size 200)")

tf.flags.DEFINE_integer("evaluate_interval",100,"Evaluate model interval(default 100)")

tf.flags.DEFINE_integer("checkpoint_interval",100,"Save Checkpoint Interval(default 100)")

tf.flags.DEFINE_integer("num_checkpoints",5,"number of checkpoints to save(default 5)")


#==============================================================================
# misc parameters 
#==============================================================================

tf.flags.DEFINE_bool("allow_soft_parameters",True,"allow soft device placement(default true)")
tf.flags.DEFINE_bool("log_device_placement",False,"Log placement of ops on devices(default false)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")

for attr,value in sorted(FLAGS.__flags.items()):
    print("%s:%s" % (attr.upper(),value))

print("\n")    



#==============================================================================
# Data Preparation
#==============================================================================

# Load data
print("Loading Data...\n")

x,y = data_util.load_data_and_labels(FLAGS.data_postive_path,FLAGS.data_negative_path)

# construct vocabulary

max_sentence_length = max([len(sent.split(" ")) for sent in x])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
x = np.array(list(vocab_processor.fit_transform(x)))

# shuffle data
np.random.seed(10)
shuffled_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffled_indices]
y_shuffled = y[shuffled_indices]

# split train-test set
# have a try with k-fold cross-validation later.

validation_set_index = -1 * int(FLAGS.validation_set_percentage * float(len(y)))
x_train,x_val = x_shuffled[:validation_set_index],x_shuffled[validation_set_index:]
y_train,y_val = y_shuffled[:validation_set_index],y_shuffled[validation_set_index:]

print("Vocabulary Size: %d" % len(vocab_processor.vocabulary_))
print("Length of train/validation set: %d , %d ." % (len(y_train),len(y_val)))


#==============================================================================
# Training
#==============================================================================






















 
 










    












































































































































































