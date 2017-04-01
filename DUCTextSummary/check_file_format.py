#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:14:27 2017
@author: chosenone
"""

from __future__ import print_function
import re


class DocumentCluster(object):
    def __init__(self,summary,sentences):
        self.summary = summary
        self.sentences = sentences

corpus_path = "/home/chosenone/download/DUC_COPRUS/duc2002.txt"

with open(corpus_path,'rb') as f:
    content = f.read()
    
    clusters = re.split(r"\r\n\r\n[A-Za-z]+",content)
    
    for i,cluster in enumerate(clusters):
        print("No.%d ..." % (i+1))
        
        document = re.split(r"[\r\n]+0",cluster)
        
        document = [sent.strip() for sent in document]
        
        summary = document[0]
        sentences = []
#==============================================================================
#         print(summary)
#==============================================================================
        
        # get the sentences for every document 
        for index in range(len(document) - 1):
               three = re.split(r"\s",document[index+1])
               sentences.append(" ".join(three[2:])) 
               print(" ".join(three[2:]))
                
        print("\n\n")
    
    print(len(re.split(r"\r\n\r\n[A-Za-z]+",content)))

