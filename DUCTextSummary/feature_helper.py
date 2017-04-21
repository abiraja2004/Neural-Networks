#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# return the feature of each sentence in format (position,avg-tf,avg-cf,number,named-entity,stop ratio)

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import re
import os
import collections
import numpy as np
import nltk
#==============================================================================
# from nltk.book import FreqDist
#==============================================================================
import time

try:
    import cPickle as pickle
except ImportError:
    import pickle    

corpus_path = "/home/chosenone/download/DUC_COPRUS/duc%s.txt"
summary_path = "/home/chosenone/download/rouge2.0-0.2/duc%s/reference/"
system_path = "/home/chosenone/download/rouge2.0-0.2/duc%s/system/"

vocabulary_path = "/home/chosenone/download/corpus/ducdata_version_0.2.txt"


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

def parseData():
    
    ALL_SENTENCES = {}
    
    ALL_SENTENCES['2001'] = []
    ALL_SENTENCES['2002'] = []
    ALL_SENTENCES['2004'] = []
    
    ALL_CLUSTERS = {} 
    for year in ['2001','2002','2004']:
    
        print("Parsing the duc%s data..." % year)
        clusters = []
        cluster_id = 1
        another_cluster = False
        FirstIn = True
        sentence_index = 0
        
        sentences = []
        cluster = {}
        
        # Load sentences
        
        for line in open(corpus_path % year,"rb"):
            if  line.strip():
                if re.match(r"^[01](\.[0-9]+)?\s+[0-9]+\s+.",line):
                    three = line.split()
                    document_id = str(three[1])
                    sentence = " ".join(three[2:])
                    sentence = _clear_str(sentence)                     
                    if not document_id in cluster.keys():
                        cluster[document_id] =  [] # sentence of document id.
                        cluster[document_id].append(sentence)
                    else:
                        cluster[document_id].append(sentence)
                        
                    ALL_SENTENCES[year].append(sentence)
                    
                    sentence_index += 1
                    
                    if another_cluster:
                        another_cluster = False
                        cluster_id += 1
                    if not FirstIn:
                        FirstIn = True
                    
                else:
        
                    if not cluster_id == 1 and FirstIn:
                        clusters.append(cluster)
                        cluster = {}
                        FirstIn = False
                        
                    another_cluster = True
                    
        clusters.append(cluster)
        
        print("Cluster Number: %d,Total sentences:%d "%(cluster_id-1,sentence_index))
        
        ALL_CLUSTERS[year] = clusters
    return ALL_CLUSTERS,ALL_SENTENCES

# return the result by the order of 2001.2002.2004
# cal Term Frequence in Every Document Cluster    
# the cal of tf may need to remove stopwords !!!
def calTF_POS(clusters,sentences):
    _TF = {}
    _POS = {}
    for year in ['2001','2002','2004']:
        cur_year_tf = []
        cur_year_pos = []
        # cal TF and POSTION
        clusters_per_year = clusters[year]
        for i in range(len(clusters_per_year)):
            all_sentences_cluster_i = []
            pos_cluster_i = []
            cluster_i = clusters_per_year[i] 
            for key in range(len(cluster_i)):
                all_sentences_cluster_i += clusters_per_year[i][str(key)]
                for index,sent in enumerate(clusters_per_year[i][str(key)]):
                    pos_cluster_i.append(index / len(clusters_per_year[i][str(key)]))
                
            freqdist = FreqDist(("  ".join(all_sentences_cluster_i)).split())
            cur_cluster_tfs = [[freqdist.freq(sample) for sample in sentence.split()] #*****
                                         for sentence in all_sentences_cluster_i]
            cur_cluster_tf =  [ sum(tfs)/len(tfs) for tfs in cur_cluster_tfs] # average-tf of sentence
            cur_year_tf += cur_cluster_tf             
            cur_year_pos += pos_cluster_i
            
        _TF[year] = cur_year_tf
        _POS[year] = cur_year_pos   
    return _TF,_POS        

def calNE_STOP_NUM(clusters,sentences):
    _NE = {}
    _STOP = {}
    _NUM = {}
    
    for year in ['2001','2002','2004']:
        cur_year_ne = []
        cur_year_num = []
        cur_year_stop = []
        
        # cal NE and NUMBER and Stop Ratio
        clusters_per_year = clusters[year]
        for i in range(len(clusters_per_year)):
            all_sentences_cluster_i = []
            ne_cluster_i = []
            num_cluster_i = []
            stop_cluster_i = []
            
            cluster_i = clusters_per_year[i] 
            for key in range(len(cluster_i)):
                for index,sent in enumerate(clusters_per_year[i][str(key)]):
                    
                    ne_cluster_i.append(1)
                    num_cluster_i.append( 1 if re.search(r"\s+[0-9]+\s+",sent) else 0)
                    stop_cluster_i.append(sum([ 1 for word in sent.split()
                                if word.lower() in nltk.corpus.stopwords.words("english")]) / len(sent))
                
            cur_year_ne += ne_cluster_i
            cur_year_num += num_cluster_i
            cur_year_stop += stop_cluster_i
            
        _NE[year] = cur_year_ne
        _NUM[year] = cur_year_num
        _STOP[year] = cur_year_stop
             
    return _NE,_STOP,_NUM         


                    
def extractFeatures():
    features = {} # shape:3 * length * (position,avg-tf,avg-cf,number,named-entity,stop ratio)
    clusters,sentences = parseData()
    tfs,poses = calTF_POS(clusters,sentences)
    nes,stops,nums = calNE_STOP_NUM(clusters,sentences)
    for year in ['2001','2002','2004']:
        print(len(tfs[year]),len(poses[year]),len(nes[year]),len(stops[year]),len(nums[year]))
        tf_year = np.reshape(tfs[year],[-1,1])
        pos_year = np.reshape(poses[year],[-1,1])
        ne_year = np.reshape(nes[year],[-1,1])
        stop_year = np.reshape(stops[year],[-1,1])
        num_year = np.reshape(nums[year],[-1,1])
        features[year] = np.concatenate([tf_year,pos_year,ne_year,stop_year,num_year],axis=1)
        print("-----------")
    
    # saving the features
    print("saving into feature.txt...")
    with open("feature.txt","wb") as f:
        pickle.dump(features,f)
        
    return features

def LoadFeatures():
    features = {}
    if not os.path.exists("feature.txt"):
        return extractFeatures()
    else:
        with open("feature.txt","rb") as f:
            features = pickle.load(f)
        return features    
    
    
begin = time.time()
features = LoadFeatures()
end = time.time()

print(len(features['2001']),(features['2001'][5][0]))
print("Time costed:%s s..." % (end - begin))
    