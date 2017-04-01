#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Fri Mar 31 20:30:58 2017
@author: chosenone

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import os
import random
import word2vec
import tensorflow as tf
import time


glove_model='/home/chosenone/download/glove.6B/glove_model.200d.2.txt'

begin = time.time()
model=word2vec.load(glove_model) #GloVe Model
end = time.time()

print("Load GloVe Model Sucessfully within %s s..." % (end - begin))                   
                                                     
CORPUS_PATH = 'corpus/'

batch_size = 1

documents = []

filenames = os.listdir(CORPUS_PATH)

random.shuffle(filenames)

for filename in filenames[:batch_size]:
    
    with open(CORPUS_PATH + filename,'rb') as f:
        
        document = {}
        
        entitymapping = {}
        
        data = f.read()
        # split into four parts
        sub_parts = re.split(r'\n\n',data)
        
        # parse to get  sentence and tag
        sentenceandtags = re.split(r'\n',sub_parts[1])
        sentences = [line[0:-4] for line in sentenceandtags]
        sentences = [re.split(r'[\s*]',sentence) for sentence in sentences]
        
        sentences = np.array([[list(model[word]) for word in sentence if word in model.vocab] for sentence in sentences])
        
        print(sentences)
        tags = np.array([ int(line[-1]) if int(line[-1]) == 1 else 0 for line in sentenceandtags])
        
        document['sentences'] = sentences
        document['tags'] = tags
        
                
        documents.append(document)
        
print((documents[0]['tags']).shape)        
print ((documents[0]['sentences'].shape))


def batch_iter(CORPUS_PATH,batch_size,nepoch,shuffle=True):
    '''
    batch reader for training set
    '''
    
    filenames = os.listdir(CORPUS_PATH)
    data_len = len(filenames)
    num_batch_per_epoch = int((data_len - 1) / batch_size) + 1
    
    for i in range(nepoch):
        
        if shuffle:
            random.shuffle(filenames)
        
        for batch_num in range(num_batch_per_epoch):
            
            documents = []
            
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size,data_len-1)
            
            for filename in filenames[start:end]:
                documents = {}
                with open(CORPUS_PATH + filename,'rb') as f:
                    
                    entitymapping = {}
                    
                    data = f.read()
                    # split into four parts
                    sub_parts = re.split(r'\n\n',data)
                    
                    # parse to get  sentence and tag
                    sentenceandtags = re.split(r'\n',sub_parts[1])
                    sentences = np.array([line[0:-4] for line in sentenceandtags])
                    sentences = np.array([re.split(r'[\s*]',sentence) for sentence in sentences])    
                    sentences = np.array([[list(model[word]) for word in sentence if word in model.vocab] for sentence in sentences])
                    tags = np.array([ int(line[-1]) if int(line[-1]) == 1 else 0 for line in sentenceandtags])
                    
                    document['sentences'] = sentences
                    document['tags'] = tags
                    document['entitymapping'] = entitymapping
                            
                    documents.append(document)
                    
            yield documents        
    