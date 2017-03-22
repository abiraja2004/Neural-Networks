#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:09:39 2017

@author: chosenone
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import word2vec

fileSegmented = '/home/chosenone/Neural-Networks/word2vec/corpus_content_segmented.txt'
modelPath = '/home/chosenone/Neural-Networks/word2vec/corpus_word_vector.bin'

word2vec.word2vec(fileSegmented,modelPath,size=300,verbose=True)


model = word2vec.load('corpus_word_vector.bin')

print(model.vectors)

print(model.vocab[1000])

indexs = model.cosine(u'我')

for index in indexs[0]:
    print(model.vocab[index])