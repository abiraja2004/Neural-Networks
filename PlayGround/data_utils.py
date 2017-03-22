#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 12:54:46 2017

@author: chosenone
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import re
import numpy as np
import os
import random

CORPUS_PATH = 'corpus/'
batch_size = 1

documents = []

filenames = os.listdir(CORPUS_PATH)

random.shuffle(filenames)

for filename in filenames[:1]:
    with open(CORPUS_PATH + filename,'rb') as f:
        document = {}
        
        data = f.read()
        # split into four parts
        sub_parts = re.split(r'\n\n',data)
        
        # parse to get  sentence and tag
        sentenceandtags = re.split(r'\n',sub_parts[1])
        sentences = [line[0:-4] for line in sentenceandtags]
    #==============================================================================
    #     sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    #==============================================================================
        sentences = [re.split(r'[\s*]',sentence) for sentence in sentences]    
        tags = [line [-1] for line in sentenceandtags]
        
        document['sentences'] = sentences
        document['tags'] = tags        
        documents.append(document)
        
print(len(documents[0]['sentences']))        
