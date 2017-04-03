#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Evaluate the CNN Model for DUC Document Summarization
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import time 
import data_helper
import datetime

from CNN4DUCSummary import CNN4DUCSummary
from tensorflow.contrib import learn

import csv

#==============================================================================
# parameters
#==============================================================================
#==============================================================================
# Data parameters
#==============================================================================
tf.flags.DEFINE_string("data_postive_path","./data/rt-polaritydata/rt-polarity.pos"
                       ,"postive data path")
tf.flags.DEFINE_string("data_negative_path","./data/rt-polaritydata/rt-polarity.neg"
                       ,"negative data path")

#==============================================================================
# evaluate parameters
#==============================================================================

tf.flags.DEFINE_integer("batch_size",64,"Batch size(default 64)")
tf.flags.DEFINE_string("checkpoint_path","","checkpoint path from training runs")
tf.flags.DEFINE_boolean("Eval_Training_Set",False,"Evaluate on the whole traing set?")

#==============================================================================
# Misc parameters
#==============================================================================

tf.flags.DEFINE_bool("allow_soft_parameters",True,"allow soft device placement(default true)")
tf.flags.DEFINE_bool("log_device_placement",False,"Log placement of ops on devices(default false)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")

for attr,value in sorted(FLAGS.__flags.items()):
    print("%s:%s" % (attr.upper(),value))

print("\n")

# Load data

if FLAGS.Eval_Training_Set:
    x_raw,y_raw = data_util.load_data_and_labels(FLAGS.data_postive_path,FLAGS.data_negative_path)
    y_test = np.argmax(y_raw,1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_raw = [1, 0]

#restore vocabulary
    
vocab_path = os.path.join(FLAGS.checkpoint_path,"..","vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

x_test = np.array(list(vocab_processor.transform(x_raw)))

print("Evaluating...\n")


#==============================================================================
# Evaluation
#==============================================================================

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

graph = tf.Graph()
with graph.as_default():
    session_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_parameters,
                                    log_device_placement=FLAGS.log_device_placement)
    
    sess  = tf.Session(config=session_config)
    with sess.as_default():
        
        saver = tf.train.import_meta_graph("%s.meta" % checkpoint_file)
        saver.restore(sess,checkpoint_file)
        
        input_x = graph.get_operation_by_name("input/x").outputs[0]
        keep_prob = graph.get_operation_by_name("input/keep_prop").outputs[0]
        
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        
        batches = data_util.batch_iter(list(x_test),FLAGS.batch_size,1,shuffle=False)
        
        
        all_predictions = []
        
        for x_test_batch in batches:
            
            feed_dict={input_x:x_test_batch,keep_prob:1.0}
            
            batch_predictions = sess.run(predictions,feed_dict)
            
            all_predictions = np.concatenate([all_predictions,batch_predictions])
            
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total Number of Test Files: %d, Accuracy: %.3f" %
                   (len(y_test),correct_predictions / float(len(y_test))))
    


# Save the predictions into a CSV file

prediction_csv = np.column_stack((np.array(x_raw),all_predictions))

csv_path = os.path.join(FLAGS.checkpoint_path,"..","predictions.csv") 

print("Saving The predictions into %s" % csv_path)

with open(csv_path,"w") as f:
    csv.writer(f).writerows(prediction_csv)
    
    
               
        
        




























   























































