#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:51:21 2017

@author: chosenone
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import numpy as np
import re
import itertools
from collections import Counter

def clear_str(string):
    '''
    Tokenized | string clean
    '''
    
    string = re.sub(r"[A-Za-z0-9(),!?\'\`]"," ",string)
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
    return string.strip().lower()
    


def load_data_and_labels(positive_data_path,negative_data_path):
    
    # load data from file and splits sentences into words and generates labels
    positive_examples = list(open(positive_data_path,"r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    
    negative_examples = list(open(negative_data_path,"r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    
    x = positive_examples + negative_examples
    x = [clear_str(s) for s in x]
    
    positive_labels = [[0,1] for pos in positive_examples]
    negative_labels = [[1,0] for neg in negative_examples]
    
    y = np.concatenate([positive_labels,negative_labels],0)
    
    return [x,y]
    
    
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
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                           
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    