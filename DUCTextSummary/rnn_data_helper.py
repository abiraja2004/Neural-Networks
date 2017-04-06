#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sat Apr  1 22:14:27 2017
@author: chosenone
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import re
import os
import csv
import collections
import numpy as np

corpus_path = "/home/chosenone/download/DUC_COPRUS/duc%s.txt"
summary_path = "/home/chosenone/download/rouge2.0-0.2/duc%s/reference/"
system_path = "/home/chosenone/download/rouge2.0-0.2/duc%s/system/"

vocabulary_path = "/home/chosenone/download/corpus/ducdata_version_0.1.txt"


rouge_1_scores_path = "Scores/results_duc%s_rouge-1.csv"
rouge_2_scores_path = "Scores/results_duc%s_rouge-2.csv"

evaluate_data_path = "/home/chosenone/Neural-Networks/DUCTextSummary/Corpus/Test/"

ALL_SENTENCES = {}

UNKNOWN = "UNKNOWN"

def LoadSentencesAndFScores():
    
    for year in ['2001','2002','2004']:
    
        print("Parse the duc%s data...\n\n" % year)
        
        with open(corpus_path % year,'rb') as f:
            content = f.read()
            clusters = re.split(r"\r\n\r\n(?=[A-Za-z]+)",content)
            
            print("Total clusters : %d\n\n" % len(clusters))
            
            total = 0
            sentences = []
            sent_index = 0
            
            for i,cluster in enumerate(clusters):
                print("No.%d ..." % (i+1))
                
                
                
                document = re.split(r"[\r\n]+(?=0.)",cluster)
                
                document = [sent.strip() for sent in document]
                
                summary = document[0]
                
                
                
                # write the gold standard summary into the reference dir of ROUGE
                if not os.path.exists(summary_path % year + ("task%d_reference" % (i+1))):
                    with open(summary_path % year + ("task%d_reference" % (i+1)),'wb') as target:
                            target.write(summary.strip()+"\r\n")
                    
                
                # get the sentences for every document 
                
                for index in range(1,len(document)):
                       
                       three = re.split(r"\s",document[index])
                       
                       sent_index += 1
                       
                       with open(system_path % year + "task%d_%s" % (i+1,sent_index),
                                 "wb") as target:
                           target.write(" ".join(three[2:])+"\r\n")
                               
                       sentences.append(" ".join(three[2:]))
                       
                       if index == len(document) - 1:
                           total += (int(three[1]) + 1)
                        
                print("\n")
            
            print("Total document: %d ,Total sentence: %d ..." % (total,sent_index))
            
            ALL_SENTENCES[year] = sentences
    

    total_sentences_train = []
    total_sentences_evaluate = []                     
    total_scores_train = []
    total_scores_evaluate = []
    
    for key in ALL_SENTENCES.iterkeys():
        
        rouge1_scores = []
        rouge2_scores = []
        scores = []
        alpha = 0.8
        
        with open(rouge_1_scores_path % key,"rb") as rouge1:
            reader = csv.reader(rouge1)
            for i,row in enumerate(reader):
                if i > 0:
                    rouge1_scores.append(row[5])
            
        with open(rouge_2_scores_path % key,"rb") as rouge2:
            reader = csv.reader(rouge2)
            for i,row in enumerate(reader):
                if i>0:
                    rouge2_scores.append(row[5])
            
        for i,sentence in enumerate(ALL_SENTENCES[key]):
            scores.append(10 * alpha * float(rouge1_scores[i]) + 10 * (1 - alpha) * float(rouge2_scores[i]))
        
        if key in ['2001','2002']:
            total_sentences_train += ALL_SENTENCES[key]
            total_scores_train += scores
        else:
            total_sentences_evaluate += ALL_SENTENCES[key]
            total_scores_evaluate += scores
                
        
    print(len(ALL_SENTENCES))
    
    print(len(ALL_SENTENCES['2001']))
    print(len(ALL_SENTENCES['2002']))
    print(len(ALL_SENTENCES['2004']))
    
    print(len(ALL_SENTENCES['2001']) + len(ALL_SENTENCES['2002'])
           + len(ALL_SENTENCES['2004']))
    
    print(len(total_sentences_train),len(total_scores_train))
    print(len(total_sentences_evaluate),len(total_scores_evaluate))
    
    

    return ((clear_sentence_array(total_sentences_train + total_sentences_evaluate)),
           total_scores_train + total_scores_evaluate,
           (clear_sentence_array(total_sentences_evaluate)),
           total_scores_evaluate)                    


def _clear_str(string):
    
    '''
    Tokenized | string clean
    '''
    string = re.sub(r"\."," . ",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]"," ",string)
    string = re.sub(r"\'s"," \'s",string)
    string = re.sub(r"\'ve"," \'ve",string)
    string = re.sub(r"n\'t"," n\'t",string)
    string = re.sub(r"\'re"," \'re",string)
    string = re.sub(r"\'d"," \'d",string)
    string = re.sub(r"\'ll"," \'ll",string)
    string = re.sub(r"!"," ! ",string)
    string = re.sub(r","," , ",string)
    string = re.sub(r"\("," \( ",string)
    string = re.sub(r"\)"," \) ",string)
    string = re.sub(r"\?"," \? ",string)
    string = re.sub(r"\s{2,}"," ",string)
    string = " " + string.strip() + " "
    return string.lower()

def clear_sentence_array(sentences):
    return [_clear_str(sentence) for sentence in sentences]
        


def build_vocab(filename):
    
    word_to_id = {}
    index = 0
    word_to_id[UNKNOWN] = index
    index += 1
          
    for line in open(filename,"rb"):
        words = line.strip().split()
        for word in words:
            if not word in word_to_id:
                word_to_id[word] = index
                index += 1          

    return word_to_id


# transform the document into id matrix, maybe there need padding or truncating
def sentence2ids(word_to_id,document,max_sent_len=200,max_num_sents=200):
    
    document_in_ids = []
    for i in range(0,max_num_sents):
        if i < len(document):
            sentenc_in_id = []
            for j in range(0,max_sent_len):
                if j < len(document[i]):
                    sentenc_in_id.append(word_to_id[document[i][j]])
                else:
                    sentenc_in_id.append(word_to_id[UNKNOWN])
                
        else:
            document_in_ids.append([word_to_id[UNKNOWN]]*max_sent_len)
    return document_in_ids


def batch_iter(data,batch_size,num_epochs,shuffle=True):
    
    """
    generate a batch iterator for dataset
    """
    
    data = np.array(data)
    data_size = len(data)
    
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
                               
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
            
        for batch_num in range(num_batches_per_epoch):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size,data_size)
            
            yield shuffled_data[start:end]
 
            
# build vocabulary first                 
word_to_id = len(build_vocab(vocabulary_path))
print(word_to_id)        

#==============================================================================
# LoadSentencesAndFScores()
#==============================================================================

# aggregateDUCdata into one file for further use        
#==============================================================================
# AggregateDucData(vocabulary_path)                
#==============================================================================


# split the DUC2004
#==============================================================================
# spliteDUC2004DataforEvaluateModel()
#==============================================================================


