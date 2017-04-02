#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:14:27 2017
@author: chosenone
"""

from __future__ import print_function
import re
import os

class DocumentCluster(object):
    def __init__(self,summary,sentences):
        self.summary = summary
        self.sentences = sentences

corpus_path = "/home/chosenone/download/DUC_COPRUS/duc%s.txt"
summary_path = "/home/chosenone/download/rouge2.0-0.2/duc%s/reference/"
system_path = "/home/chosenone/download/rouge2.0-0.2/duc%s/system/"


ALL_SENTENCES = {}

def ParseSentenceAndCalculateFScore():
    
    for year in ['2001','2002']:
    
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
                         
    print(len(ALL_SENTENCES))
    
    print(len(ALL_SENTENCES['2001']))
    print(len(ALL_SENTENCES['2002']))

def AggregateDucData(target_path):
    
    ParseSentenceAndCalculateFScore()
    for key in ALL_SENTENCES.iterkeys():
        sentences = ALL_SENTENCES[key]
        print(sentences[0])
        with open(target_path,'a') as target:
            for sentence in sentences:
                target.write(sentence+" ")
                
                
AggregateDucData("/home/chosenone/download/corpus/ducdata.txt")                
            
                 
        
        
