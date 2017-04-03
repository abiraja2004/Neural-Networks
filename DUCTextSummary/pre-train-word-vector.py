#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:21:32 2017

@author: chosenone
"""

from __future__ import print_function
import word2vec

filePath = "/home/chosenone/download/corpus/ducdata.txt"
      

# train word vector

word2vec.word2vec(filePath,'duc_corpus_word_vector.bin',min_count=0,size=200,verbose=True)        


#==============================================================================
# model = word2vec.load('duc_corpus_word_vector.bin')
# 
# print(len(model.vocab))
# indexs,_ = model.cosine("you")
# 
# for index in indexs:
#     print(model.vocab[index])
#==============================================================================
    
    
