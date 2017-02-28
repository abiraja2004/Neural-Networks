#coding=utf8
"""
@author: Lebron.Ran
@file: preprocessData_checkRNN.py
@time: 2017/2/27 0027-23:51
"""
import csv
import itertools
import operator
import numpy as np
import nltk
from RNN_numpy import RNN_numpy
import sys
from datetime import datetime

import matplotlib.pyplot as plt

vocabulary_size = 8000
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
SENTENCE_START = 'SENTENCE_START'
SENTENCE_END = 'SENTENCE_END'

# pre-processing the training data.

print 'Reading the CSV files ...'

with open(r'data/reddit-comments-2015-08.csv','rb') as f:
    reader  =csv.reader(f,skipinitialspace=True)
    reader.next()
    #split all of comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf8').lower()) for x in reader])
    # append the SENTENCE_START and SENTENCE_END for all sentences
    sentences = ["%s %s %s" % (SENTENCE_START,x,SENTENCE_END) for x in sentences]
    print 'Parsed %d sentences.' % len(sentences)

# make sentence tokenized
tokenized_sentences = [nltk.word_tokenize(sen) for sen in sentences]

# count TF
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

print 'Have found %d unique words.' % len(word_freq.items())

# get the most common words and build the index_to_word and word_to_index vector
voca_freq = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in voca_freq]
index_to_word.append(UNKNOWN_TOKEN)
word_to_index = dict([(w,i) for (i,w) in enumerate(index_to_word)])

print 'Using vocabulary size: %d' % vocabulary_size
print 'The least frequent word in our train set is "%s" and appear %d times' % (voca_freq[-1][0],\
                                                                              voca_freq[-1][1])

# replace all the words not in our vocabulary with UNKNOWN_TOKEN
for i,sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

# to check the performance of our pre-processing work

print 'Example sentence: "%s" \n'%sentences[0]
print 'Example sentence after pre-processing: "%s"' % tokenized_sentences[0]

#creat training data set

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]]for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# check
x_example = X_train[17]
y_example = y_train[17]
print 'x:\n%s\n%s' % (" ".join([index_to_word[x] for x in x_example]),x_example)
print 'y:\n%s\n%s' % (" ".join([index_to_word[y] for y in y_example]),y_example)



# build a vanilla RNN model
# some scripts to check every part of RNN when implementing it.


# check forward propagation and predict function

np.random.seed(10)
model =  RNN_numpy(vocabulary_size)
y,s = model.forward_propagation(X_train[10])
print y.shape
print np.matrix(s[0]).shape
print y

predictions = model.predict(X_train[10])
print predictions.shape
print predictions

# check our loss function with only 1000 samples to save time.

print "Expected Loss for random predictions: %f\n" % np.log(vocabulary_size)
print "Actual Loss : %f\n" % model.calculate_loss(X_train[:1000],y_train[:1000])










