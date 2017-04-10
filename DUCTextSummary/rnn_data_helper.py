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

vocabulary_path = "/home/chosenone/download/corpus/ducdata_version_0.2.txt"


rouge_1_scores_path = "Scores/results_duc%s_rouge-1.csv"
rouge_2_scores_path = "Scores/results_duc%s_rouge-2.csv"

evaluate_data_path = "/home/chosenone/Neural-Networks/DUCTextSummary/Corpus/Test/"

ALL_CLUSTERS = {}

UNKNOWN = "UNKNOWN"

def LoadSentencesAndFScores():
    
    for year in ['2001','2002','2004']:
    
        print("Parse the duc%s data..." % year)
        
        clusters = []
        cluster_id = 1
        another_cluster = False
        FirstIn = True
        sentence_index = 0
        
        sentences = []
        cluster = {}
        score = []
        
        
        rouge1_scores = []
        rouge2_scores = []
        scores = []
        alpha = 0.5
        
        
        # Load scores
        
        with open(rouge_1_scores_path % year,"rb") as rouge1:
            reader = csv.reader(rouge1)
            for i,row in enumerate(reader):
                if i > 0:
                    rouge1_scores.append(row[5])
            
        with open(rouge_2_scores_path % year,"rb") as rouge2:
            reader = csv.reader(rouge2)
            for i,row in enumerate(reader):
                if i>0:
                    rouge2_scores.append(row[5])
            
        scores = (10 * alpha * np.array(map(float,rouge1_scores)) 
                    + 10 * (1 - alpha) * np.array(map(float,rouge2_scores)))
        
               
        # Load sentences
        
        for line in open(corpus_path % year,"rb"):
            if  line.strip():
                if re.match(r"^[01](\.[0-9]+)?\s+[0-9]+\s+.",line):
                    three = line.split()
                    what_the_fuck = three[0]
                    document_id = int(three[1])
                    sentence = " ".join(three[2:])
                    
                    sentences.append(sentence)
                    score.append(scores[sentence_index])
                    
                    sentence_index += 1
                    
                    if another_cluster:
                        another_cluster = False
                        cluster_id += 1
                    if not FirstIn:
                        FirstIn = True
                    
                else:
        
                    if not cluster_id == 1 and FirstIn:
                        cluster['id'] = cluster_id - 1
                        cluster['score']  = score
                        cluster['sentence'] = sentences
                        clusters.append(cluster)
                        cluster = {}
                        score = []
                        sentences = []
                        FirstIn = False
                        
                    another_cluster = True
                    
        cluster['id'] = cluster_id - 1
        cluster['score']  = score
        cluster['sentence'] = sentences
        score = []
        sentences = []            
        clusters.append(cluster)
        
        print("Cluster Number: %d,Total sentences:%d "%(cluster_id-1,sentence_index))
        
        
        
        print("-------split line---------")
        ALL_CLUSTERS[year] = clusters
    

        
    print(len(ALL_CLUSTERS))
    
    print(len(ALL_CLUSTERS['2001']))
    print(len(ALL_CLUSTERS['2002']))
    print(len(ALL_CLUSTERS['2004']))
    
    print(len(ALL_CLUSTERS['2001']) + len(ALL_CLUSTERS['2002'])
           + len(ALL_CLUSTERS['2004']))
    
    print("------split line------")
    total = 0
    more_than_400_sentences = 0
    length = 550
    for year in ['2001','2002','2004']:
        for cluster in ALL_CLUSTERS[year]:
            total += 1
            if len(cluster['sentence']) > length:
                more_than_400_sentences += 1
    print(ALL_CLUSTERS['2001'][0]['score'][0])         
    print("Total Clusters: %d,Longer than %d : %d,i.e. %.3f%%" % (total,length,
                more_than_400_sentences
                ,more_than_400_sentences / total * 100))   
    
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
def sentence2ids(word_to_id,document,max_sent_len=60,max_num_sents=550):
    
    document_in_ids = []
    for i in range(0,max_num_sents):
        if i < len(document):
            sentenc_in_id = []
            for j in range(0,max_sent_len):
                if j < len(document[i]):
                    if document[i][j] in word_to_id:
                        sentenc_in_id.append(word_to_id[document[i][j]])
                    else:
                        sentenc_in_id.append(0)
                else:
                    sentenc_in_id.append(word_to_id[UNKNOWN])
            document_in_ids.append(sentenc_in_id)        
                
        else:
            document_in_ids.append([word_to_id[UNKNOWN]]*max_sent_len)
    return document_in_ids


def batch_iter(num_epochs,shuffle=True):
    '''
    return one document in numpy-array format during every iteration
    '''
    # build vocabulary    
    word_to_id = build_vocab(vocabulary_path)
    # load sentences  and scores
    LoadSentencesAndFScores()
    
    for year in ALL_CLUSTERS.iterkeys():
        cur_duc = ALL_CLUSTERS[year]
        for i in range(len(cur_duc)):
          sentences_cleared = clear_sentence_array(ALL_CLUSTERS[year][i]['sentence'])
          
          sentences_in_word = [sentence.split() for sentence in sentences_cleared]
          scores = ALL_CLUSTERS[year][i]['score']
          
          sentences_in_id = sentence2ids(word_to_id,sentences_in_word)
          
          # clip scores
          document_len = len(sentences_in_id)
          
          if len(scores) < document_len:
              scores += [0.0] * (document_len - len(scores))
          else:
              scores = scores[:document_len]
          
          sentences_in_id = np.array(sentences_in_id)
          scores = np.reshape(np.array(scores),(-1,1))
          
          yield sentences_in_id,scores
    
            
# build vocabulary first                 
#==============================================================================
# word_to_id = len(build_vocab(vocabulary_path))
# print(word_to_id)        
#==============================================================================

# just for debug
verbose = True
i = 0
for a,b in batch_iter(1):
    if i==1:
       print(np.array(a).shape)
       print(np.reshape(np.array(b),(-1,1)).shape)
       verbose = False
    print(len(a))
    print(len(b))
    print("+++")
    i += 1
    
print(i)
def TestReForDUC():
    i = 0
    max_len = 0
    reference_num = 0
    max_reference_num = -1
    sent_index = 0
    more_than_200_num = 0
    total_num = 0
    for line in open(corpus_path % "2004","rb"):
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
                
                sent_index += 1
                reference_num = 0
            else:
                print(sent_index)
                if sent_index > 400:
                    more_than_200_num += 1
                total_num += 1    
                sent_index = 0
                reference_num += 1
            
    print(max_len,i,max_reference_num,sent_index,more_than_200_num,total_num / max_reference_num)


#==============================================================================
# TestReForDUC()
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


