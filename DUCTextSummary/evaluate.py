#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Evaluate the CNN Model for DUC Document Summarization
Output The Summarization to the SYSTEM_SUMMARY_PATH and Evaluate it using ROUGE Toolkit

"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import time 
import evaluate_data_helper
import datetime

from CNN4DUCSummary import CNN4DUCSummary
from tensorflow.contrib import learn

import csv

#==============================================================================
# parameters
#==============================================================================

#==============================================================================
# evaluate parameters
#==============================================================================

tf.flags.DEFINE_integer("batch_size",64,"Batch size(default 64)")
tf.flags.DEFINE_string("checkpoint_path","/home/chosenone/Neural-Networks/DUCTextSummary/runs/1491402480/checkpoints"
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



def CosineSimilarity(a,b):
    dot_product = 0.0  
    normA = 0.0  
    normB = 0.0  
    
    for a,b in zip(a,b):  
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    
    if normA == 0.0 or normB==0.0:  
        return None  
    else:  
        return dot_product / ((normA*normB)**0.5)

def CalSentenceVector(word_embeddings,sentence):
    embeddings = np.array(word_embeddings)
    sentences = embeddings[sentence[:50]]
    return np.mean(sentences,axis=0)
    
                    
verbose = True
# load data
for dataset in evaluate_data_helper.batch_iter_evaluate_data():
    
    sentences = []
    
    cluster_num = 0
    for key in dataset.iterkeys():
        cluster_num = int(key)
        sentences = dataset[key]
    
    print("Generating Summary for Cluster No.%d..." % cluster_num)

    #restore vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_path,"..","vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    
    x_test = np.array(list(vocab_processor.transform(sentences)))
    
    print(np.shape(x_test))
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
            
            
            get_scores = graph.get_operation_by_name("output/scores").outputs[0]
            get_sentences = graph.get_operation_by_name("sentences").outputs[0]
            
            embeddings = graph.get_operation_by_name("embedding/embedding").outputs[0]            
            
            feed_dict={input_x:x_test,keep_prob:1.0}
            
            word_embeddings,scores,sentences_representation = sess.run([embeddings,get_scores,get_sentences],feed_dict)
            
            enheng = zip(scores,range(len(scores)))
            sortedEnheng = sorted(enheng,key=lambda x:-float(x[0]))
            
            # generate summary
            
            length = 0
            sentences_to_write = []
            indexs = []
            for i in range(len(sortedEnheng)):
                if length < 665:
                    
                    canbeAdded = True
                    
                    for j,sentence in enumerate(sentences_to_write):
                        
                        if (CosineSimilarity(sentences_representation[indexs[j]]
                            ,sentences_representation[sortedEnheng[i][1]])) > 1:
                            print(CosineSimilarity(sentences_representation[indexs[j]]
                            ,sentences_representation[sortedEnheng[i][1]]))
                            canbeAdded = False
                            break
                        
                    if canbeAdded:
                        
                        length += len(bytes(sentences[sortedEnheng[i][1]]))
                        sentences_to_write.append(sentences[sortedEnheng[i][1]])
                        indexs.append(sortedEnheng[i][1])
                else:
                    break
            
            with open(system_summary_path + ("task%d_1" % (cluster_num+1)),"wb") as target:
                target.writelines(sentences_to_write)
                    
            print(sortedEnheng[i][0],sortedEnheng[i][1])
            print(len(scores),len(sentences_representation))
            
            
                
