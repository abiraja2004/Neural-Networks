#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:53:19 2017
@author: chosenone

Train CNN for Text Classification


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

tf.flags.DEFINE_float("learning_rate",0.001,"learning rate(default 0.001)")

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

x_data,y = data_util.load_data_and_labels(FLAGS.data_postive_path,FLAGS.data_negative_path)

# construct vocabulary

max_sentence_length = max([len(sent.split(" ")) for sent in x_data])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
x = np.array(list(vocab_processor.fit_transform(x_data)))

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

print("Vocabulary Size: %s" % len(vocab_processor.vocabulary_._mapping))
print("Length of train/validation set: %d , %d ." % (len(y_train),len(y_val)))


#==============================================================================
# Training
#==============================================================================

with tf.Graph().as_default():
    session_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_parameters,
                                    log_device_placement=FLAGS.log_device_placement)
    
    sess = tf.Session(config=session_config)
    with sess.as_default():
        
        cnn = CNN4Text(sequence_length=x_train.shape[1],
                       num_classes=y_train.shape[1],
                       vocab_size=len(vocab_processor.vocabulary_),
                       embedding_size=FLAGS.embedding_size,
                       filter_sizes=list(map(int,FLAGS.filter_sizes.split(","))),
                       num_filters=FLAGS.num_filters,
                       l2_reg_lambda=FLAGS.l2_reg_lambda)
        
        # the detail of train procedure
        
        global_step = tf.Variable(0,name="global_step",trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn._loss)
        train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)
        
        grad_summaries = []
        
        for g,v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("%s/grad/hist" % v.name,g)
                sparsity_summary = tf.summary.scalar("%s/grad/hist" % v.name,tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summayies_merged = tf.summary.merge(grad_summaries)

        # output path of summary
        timestamp = str(int(time.time()))
        output_path = os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
        
        print("Writing into Output Path: %s ..." % output_path)
        
        # summary for loss and accuracy
        
        loss_summary = tf.summary.scalar("loss",cnn._loss)
        acc_summary = tf.summary.scalar("accuracy",cnn._accuracy)
        
        # train summaries
        train_summary_op = tf.summary.merge([loss_summary,acc_summary,grad_summayies_merged])
        train_summary_path = os.path.join(output_path,"summary","train")
        train_summary_writer = tf.summary.FileWriter(train_summary_path,sess.graph)
        
        #validation summaries
        validation_summary_op = tf.summary.merge([loss_summary,acc_summary])
        validation_summary_path = os.path.join(output_path,"summary","validation")
        validation_summary_writer = tf.summary.FileWriter(validation_summary_path,sess.graph)
        
        
        checkpoint_path = os.path.abspath(os.path.join(output_path,"checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_path,"model")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)    
        
        
        # save vocabulary
        vocab_processor.save(os.path.join(output_path,"vocab"))
        
        sess.run(tf.global_variables_initializer())
        
        # a single training step
        
        def train_step(x_batch,y_batch,writer=None):
            '''
            a single training step
            '''
            feed_dict = {cnn._input_x:x_batch,
                         cnn._input_y:y_batch,
                         cnn._keep_prob:FLAGS.keep_prob}
            _,step,summaries,loss,accuracy = sess.run([train_op,global_step,train_summary_op
                      ,cnn._loss,cnn._accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
            print("%s: Step: %d,Loss: %.4f,Accuracy: %.4f" % (time_str,step,loss,accuracy))
            
            if writer:
                writer.add_summary(summaries,step)
                
            
        # a single validation step    
        def validation_step(x_batch,y_batch,writer=None):
            '''
            a single training step
            '''
            feed_dict = {cnn._input_x:x_batch,
                         cnn._input_y:y_batch,
                         cnn._keep_prob:1.0} # for evaluation
            step,summaries,loss,accuracy = sess.run([global_step,validation_summary_op
                      ,cnn._loss,cnn._accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
            print("%s: Step: %d,Loss: %.4f,Accuracy: %.4f" % (time_str,step,loss,accuracy))
            
            if writer:
                writer.add_summary(summaries,step)
        
        # generates batches
        
        batches = data_util.batch_iter(list(zip(x_train,y_train)),FLAGS.batch_size,FLAGS.num_epochs)
        
        #Training Loop
        
        for batch in batches:
            x_batch,y_batch = zip(*batch)
            train_step(x_batch,y_batch,writer=train_summary_writer)
            current_step = tf.train.global_step(sess,global_step)
            if current_step % FLAGS.evaluate_interval == 0:
                print("Evaluation:\n")
                validation_step(x_batch,y_batch,writer=validation_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_interval == 0:
                path = saver.save(sess,checkpoint_prefix,global_step=current_step)
                print("Saved the model checkpoint to %s " % path)
            
        
        
        
        
        
        
        
                
                
                
        
        
        
        




















 
 










    












































































































































































