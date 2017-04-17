#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Evaluate the GRU Model for DUC Document Summarization
Output The Summarization into the SYSTEM_SUMMARY_PATH and Evaluate it using ROUGE Toolkit

"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import time 
import rnn_data_helper
import datetime
import re


from CNN4DUCSummary import CNN4DUCSummary
from tensorflow.contrib import learn

import csv

#==============================================================================
# parameters
#==============================================================================

#==============================================================================
# evaluate parameters
#==============================================================================
tf.flags.DEFINE_string("checkpoint_path","/home/chosenone/Neural-Networks/DUCTextSummary/runs/1492337378/checkpoints"
                       ,"checkpoint path from training runs")
tf.flags.DEFINE_boolean("Eval_Training_Set",False,"Evaluate on the whole traing set?")

#==============================================================================
# Misc parameters
#==============================================================================

tf.flags.DEFINE_bool("allow_soft_parameters",True,"allow soft device placement(default true)")
tf.flags.DEFINE_bool("log_device_placement",False,"Log placement of ops on devices(default false)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


system_summary_path = "/home/chosenone/download/rouge2.0-0.2/duc2004-evaluate/system/"

print("\nParameters:")

for attr,value in sorted(FLAGS.__flags.items()):
    print("%s:%s" % (attr.upper(),value))

print("\n")

verbose = True

#==============================================================================
# Evaluation
#==============================================================================



graph = tf.Graph()

with graph.as_default():
    session_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_parameters,
                                    log_device_placement=FLAGS.log_device_placement)
    
    sess  = tf.Session(config=session_config)
    
    with sess.as_default():
        

        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        
        saver = tf.train.import_meta_graph("%s.meta" % checkpoint_file)
        saver.restore(sess,checkpoint_file)
        
        input_x = graph.get_operation_by_name("input/input_x").outputs[0]
        keep_prob = graph.get_operation_by_name("input/keep_prob").outputs[0]
        
        
        get_scores = graph.get_operation_by_name("prediction/scores").outputs[0]
        
        embeddings = graph.get_operation_by_name("embedding/embedding-fine-tuned").outputs[0]            
        
        for index,(sentences,_,sentences_raw) in enumerate(rnn_data_helper.evaluate_data()):
    
            
            print("Generating Summary for Cluster No.%d..." % (index+1))

            feed_dict={input_x:sentences,keep_prob:1.0}
            
            word_embeddings,scores = sess.run([embeddings,get_scores],feed_dict)
            
            enheng = zip(scores,range(len(scores)))
            sortedEnheng = sorted(enheng,key=lambda x:-float(x[0]))
            
            # generate summary,meanwhile check the 'best' simliarity.
            
            
            length = 0
            sentences_to_write = []
            indexs = []
            
            for i in range(len(sortedEnheng)):
                if length < 665:
                    if not (sentences_raw[sortedEnheng[i][1]] in  sentences_to_write):
                        length += len(bytes(sentences_raw[sortedEnheng[i][1]]))
                        sentences_to_write.append(sentences_raw[sortedEnheng[i][1]])
                        indexs.append(sortedEnheng[i][1])
                else:
                    break
            
            with open(system_summary_path + ("task%d_GRU" % (index+1)),"wb") as target:
                for sentence_to_write in sentences_to_write:
                    target.write(sentence_to_write+"\r\n")
            
   