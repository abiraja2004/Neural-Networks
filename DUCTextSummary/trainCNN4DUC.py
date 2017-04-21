#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:33:25 2017
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
import data_helper
import re
import word2vec
import random
from random import shuffle

from CNN4DUCSummary import CNN4DUCSummary
from tensorflow.contrib import learn


#==============================================================================
# parameters
#==============================================================================

#==============================================================================
# data parameters
#==============================================================================

tf.flags.DEFINE_float("validation_set_percentage",0.05,
        "the percentage of training examples that will be used for validation set")

#==============================================================================
# model hyperparameters
#==============================================================================

tf.flags.DEFINE_float("learning_rate",0.001,"learning rate(default 0.001)")

tf.flags.DEFINE_integer("embedding_size",25,"the size of word embeeding (default 25)")

tf.flags.DEFINE_integer("num_filters",30,"the number of filters for each filter size(default 30)")

tf.flags.DEFINE_string("filter_sizes","1,2,3,4","comma-separated filter sizes(default 1~6)")

tf.flags.DEFINE_float("keep_prob",0.5,"the probability used for dropout(default 0.5)")

tf.flags.DEFINE_float("l2_reg_lambda",0,"the l2 regularization lambda(default 0)")



#==============================================================================
# train parameters
#==============================================================================

tf.flags.DEFINE_integer("batch_size",256,"Batch size (default size 64)")

tf.flags.DEFINE_integer("num_epochs",30,"Epoch sizes(default size 200)")

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
# Load Pre-Trained Model 
begin = time.time()
model = word2vec.load('duc_corpus_word_vector.bin')
end = time.time()

print("Pre-trained word vector loaded in %.3f s..." % (end - begin))


# Load data
print("Loading Data...\n")

x_data,y,features_train,x_validation_raw,y_validation_raw,features_val = data_helper.LoadSentencesAndFScores()

# just for debug
x_validation = []
y_validation = []
features_validation = []
indexs = range(len(x_validation_raw))
shuffle(indexs)
for i in indexs:
    x_validation.append(x_validation_raw[i])
    y_validation.append(y_validation_raw[i])
    features_validation.append(features_val[i])

# construct vocabulary
x_train_len = len(y)
x_val_len = len(y_validation)

sentence_array = [re.split(r"\s+",sent.strip()) for sent in x_data]

vocab_processor = learn.preprocessing.VocabularyProcessor(60)
x = np.array(list(vocab_processor.fit_transform(x_data + x_validation)))
y = np.reshape(np.array(y + y_validation[:5000]),[-1,1])
y_validation = np.reshape(np.array(y_validation),[-1,1])


print(len(vocab_processor.vocabulary_))
print(len(vocab_processor.vocabulary_._reverse_mapping))
print(len(y))

# pre-trained word vector
word_embedding = []

for i in range(len(vocab_processor.vocabulary_)):
    cur_word = vocab_processor.vocabulary_._reverse_mapping[i]
    if cur_word in model.vocab:
        word_embedding.append(list(model[cur_word]))
    else:
        word_embedding.append([random.uniform(-0.5,0.5)] * FLAGS.embedding_size)

# shuffle data
np.random.seed(10)
shuffled_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[:x_train_len + 5000][shuffled_indices]
y_shuffled = y[shuffled_indices]

# split train-test set
# have a try with k-fold cross-validation later.

validation_set_index = -1000

x_train,x_val = x_shuffled,x[validation_set_index:]
y_train,y_val = y_shuffled,y_validation[validation_set_index:]
features_train = np.concatenate([features_train,features_validation[:5000]],axis=0)
features_validation = features_validation[-1000:]

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
        
        cnn = CNN4DUCSummary(sequence_length=x_train.shape[1],
                       num_classes=y_train.shape[1],
                       vocab_size=len(vocab_processor.vocabulary_),
                       embedding_size=FLAGS.embedding_size,feature_size=np.shape(features_train)[1],
                       filter_sizes=list(map(int,FLAGS.filter_sizes.split(","))),
                       num_filters=FLAGS.num_filters,fine_tune=True,word_embedding=word_embedding,
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
        
        # train summaries
        train_summary_op = tf.summary.merge([loss_summary,grad_summayies_merged])
        train_summary_path = os.path.join(output_path,"summary","train")
        train_summary_writer = tf.summary.FileWriter(train_summary_path,sess.graph)
        
        #validation summaries
        validation_summary_op = tf.summary.merge([loss_summary])
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
        
        def train_step(x_batch,y_batch,features_batch,writer=None):
            '''
            a single training step
            '''
            try:
                feed_dict = {cnn._input_x:x_batch,
                             cnn._input_y:y_batch,
                             cnn._features:features_batch,
                             cnn._keep_prob:FLAGS.keep_prob}
                _,step,summaries,loss = sess.run([train_op,global_step,train_summary_op
                          ,cnn._loss],feed_dict)
                time_str = datetime.datetime.now().isoformat()
                
                print("%s: Step: %d,Loss: %.4f" % (time_str,step,loss))
                
                if writer:
                    writer.add_summary(summaries,step)
            except BaseException,e:
                pass
                
            
        # a single validation step    
        def validation_step(x_batch,y_batch,features_batch,writer=None):
            '''
            a single training step
            '''
            feed_dict = {cnn._input_x:x_batch,
                         cnn._input_y:y_batch,
                         cnn._features:features_batch,
                         cnn._keep_prob:1.0} # for evaluation
            step,summaries,loss = sess.run([global_step,validation_summary_op
                      ,cnn._loss],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
            print("%s: Step: %d,Loss: %.4f" % (time_str,step,loss))
            
            if writer:
                writer.add_summary(summaries,step)
        
        # generates batches
        
        batches = data_helper.batch_iter(list(zip(x_train,y_train,features_train)),FLAGS.batch_size,FLAGS.num_epochs)
        
        #Training Loop
        
        for batch in batches:
            x_batch,y_batch,features_batch = zip(*batch)
            train_step(x_batch,y_batch,features_batch,writer=train_summary_writer)
            current_step = tf.train.global_step(sess,global_step)
            
            if current_step % FLAGS.evaluate_interval == 0:
                
                print("##############\nEvaluation:\n")
                validation_step(x_val,y_val,features_validation,writer=validation_summary_writer)
                print("##############")
                
            if current_step % FLAGS.checkpoint_interval == 0:
                path = saver.save(sess,checkpoint_prefix,global_step=current_step)
                print("Saved the model checkpoint to %s " % path)