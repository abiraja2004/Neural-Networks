#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:33:40 2017

@author: chosenone
"""
import jieba
import word2vec

filePath = "/home/chosenone/download/corpus/ducdata.txt"

fileSegmented = './corpus_content_segmented.txt'


# helper function

def printList(l):
    for word in l:
        print l


# read file by line

print 'Reading the files...'

fileRaws = []

with open(filePath) as f:
    for line in f:
        fileRaws.append(line)


print 'Segmenting the words...'
       
# segment word with jieba

wordSegment = []

for i in fileRaws[:10000]:
    
    wordSegment.append(' '.join(list(jieba.cut(i[9:-11],cut_all=False))))
    
# check the effect of word-segment
printList(wordSegment[10])

print 'Saving the segmented result into target files...'
        
# save the segmented word into files

with open(fileSegmented,'wb') as tf:
    for i in wordSegment:
        tf.write(i.encode('utf-8'))
        tf.write('\n')
        
      

# train word vector

#==============================================================================
# word2vec.word2vec('corpus_content_segmented.txt','corpus_word_vector.bin',
#                   size=128,verbose=True)        
#==============================================================================



        






        







