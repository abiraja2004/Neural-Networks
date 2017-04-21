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
import feature_helper

corpus_path = "/home/chosenone/download/DUC_COPRUS/duc%s.txt"
summary_path = "/home/chosenone/download/rouge2.0-0.2/duc%s/reference/"
system_path = "/home/chosenone/download/rouge2.0-0.2/duc%s/system/"

vocabulary_path = "/home/chosenone/download/corpus/ducdata_version_0.2.txt"

rouge_1_scores_path = "Scores/results_duc%s_rouge-1.csv"
rouge_2_scores_path = "Scores/results_duc%s_rouge-2.csv"

evaluate_data_path = "/home/chosenone/Neural-Networks/DUCTextSummary/Corpus/Test/"

ALL_SENTENCES = {}


def LoadSentencesAndFScores():
    
    for year in ['2001','2002','2004']:
    
        print("Parse the duc%s data..." % year)
        
        cluster_id = 1
        
        another_cluster = False
        
        sentence_index = 0
        
        refrence_id = 1
        
        sentences = []
        
        reference_num = 0
        
        for line in open(corpus_path % year,"rb"):
            if  line.strip():
                if re.match(r"^[01](\.[0-9]+)?\s+[0-9]+\s+.",line):
                    three = line.split()
                    what_the_fuck = three[0]
                    document_id = int(three[1])
                    sentence = " ".join(three[2:])
                    
                    sentences.append(sentence)
                    sentence_index += 1
                    
                    cur_cluster_id = cluster_id - 1 if not another_cluster else cluster_id
                    
                    if another_cluster:
                        another_cluster = False
                        cluster_id += 1
                        
                    reference_num = 0
                else:
        
                    reference_num += 1
                    
                    another_cluster = True
                    
        print("Cluster Number: %d,Total sentences:%d \n\n"%(cluster_id-1,sentence_index))
       
                        
        ALL_SENTENCES[year] = sentences
            
    total_sentences_train = []
    total_sentences_evaluate = []                     
    total_scores_train = []
    total_scores_evaluate = []
    
    for key in ['2001','2002','2004']:
        
        rouge1_scores = []
        rouge2_scores = []
        scores = []
        alpha = 0.8
        
        with open(rouge_1_scores_path % key,"rb") as rouge1:
            reader = csv.reader(rouge1)
            for i,row in enumerate(reader):
                if i > 0:
                    rouge1_scores.append(row[3])
            
        with open(rouge_2_scores_path % key,"rb") as rouge2:
            reader = csv.reader(rouge2)
            for i,row in enumerate(reader):
                if i>0:
                    rouge2_scores.append(row[3])
            
        for i,sentence in enumerate(ALL_SENTENCES[key]):
            scores.append(10 * alpha * float(rouge1_scores[i]) + 10 * (1 - alpha) * float(rouge2_scores[i]))
        
        if key in ['2001','2002']:
            total_sentences_train += ALL_SENTENCES[key]
            total_scores_train += scores
        else:
            total_sentences_evaluate += ALL_SENTENCES[key]
            total_scores_evaluate += scores
                
    # Load features
    features = feature_helper.LoadFeatures()
    features_train = np.concatenate([features['2001'],features['2002']],axis=0)
    features_evaluate = features['2004']    
    return ((clear_sentence_array(total_sentences_train)),
           total_scores_train,features_train,
           (clear_sentence_array(total_sentences_evaluate)),
           total_scores_evaluate,features_evaluate)                    

def AggregateDucData(target_path):
    
    # target_path stores the corpus for training word vector
    # target_line_splited_path stores the same thing but by lines
    
    LoadSentencesAndFScores()
    
    exist = not os.path.exists(target_path)
    
    for key in ALL_SENTENCES.iterkeys():
        
        sentences = ALL_SENTENCES[key]
        
        if exist:
            with open(target_path,"a") as target:
                for sentence in clear_sentence_array(sentences):
                    target.write(sentence+"\r\n")

def spliteDUC2004DataforEvaluateModel():
    
    for year in ['2004']:
    
        print("Parse the duc%s data..." % year)
        
        clusters = []
        cluster_id = 1
        another_cluster = False
        FirstIn = True
        sentence_index = 0
        
        sentences = []
        
        # Load sentences
        
        for line in open(corpus_path % year,"rb"):
            if  line.strip():
                if re.match(r"^[01](\.[0-9]+)?\s+[0-9]+\s+.",line):
                    three = line.split()
                    what_the_fuck = three[0]
                    document_id = int(three[1])
                    sentence = " ".join(three[2:])
                    
                    sentences.append(sentence)
                    sentence_index += 1
                    
                    if another_cluster:
                        another_cluster = False
                        cluster_id += 1
                    if not FirstIn:
                        FirstIn = True
                    
                else:
        
                    if not cluster_id == 1 and FirstIn:
                        with open(evaluate_data_path + "cluster%d" % (cluster_id - 1),"wb") as target:
                             for sentence in sentences:
                                 target.write(sentence+"\r\n")
                        sentences = []
                        FirstIn = False
                        
                    another_cluster = True
                    
        with open(evaluate_data_path + "cluster%d" % (cluster_id - 1),"wb") as target:
            for sentence in sentences:
                target.write(sentence+"\r\n\r\n")
        print("Cluster Number: %d,Total sentences:%d "%(cluster_id-1,sentence_index))
        
def _clear_str(string):
    
    '''
    Tokenized | string clean
    '''
    string = re.sub(r"\."," . ",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ",string)
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
        

def read_words(filename):
    with open(filename,"r") as f:
        return f.read().split()

def build_vocab(filename):
    
    data = read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(),key=lambda x:(-x[1],x[0]))
    
    words,_ = list(zip(*count_pairs))
    word_to_id = dict(zip(words,range(len(words))))
    
    return word_to_id


def sentence2ids(sentences):
    
    word_to_id = build_vocab(vocabulary_path)
    
    return len(word_to_id),[[word_to_id[word] for word in re.split(r"\s+",sent.strip())] for sentence in sentences]


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
 
def TestReForDUC():
    i = 0
    max_len = 0
    reference_num = 0
    max_reference_num = -1
    for line in open(corpus_path % "2002","rb"):
        if  line.strip():
            if re.match(r"^[01](\.[0-9]+)?\s+[0-9]+\s+.",line):
                three = line.split()
                cur_len = len(three) - 2
                sentence = " ".join(three[2:])
                i += 1
                if cur_len > 60:
                    max_len += 1
    #==============================================================================
    #                 print(line)
    #                 print("++++")
    #==============================================================================
                if reference_num > max_reference_num:
                    max_reference_num = reference_num
                
                reference_num = 0
            else:
    
                print(line)
                print("+++")
                reference_num += 1
                print(reference_num)
            
    print(max_len,i,max_reference_num)            
# build vocabulary first                 
#==============================================================================
# word_to_id = build_vocab(vocabulary_path) 
# print(len(word_to_id))       
# 
#==============================================================================
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

