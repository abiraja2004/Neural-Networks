#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon Apr  3 14:27:41 2017
@author: chosenone
"""

from __future__ import print_function
from __future__ import absolute_import

import os
import re

evaluate_data_path = "/home/chosenone/Neural-Networks/DUCTextSummary/Corpus/Test/"

    
def batch_iter_evaluate_data():
    for i,filename in enumerate(os.listdir(evaluate_data_path)):
        with open(evaluate_data_path + filename) as target:
            data = target.read()
            key = re.match(r"(cluster)([0-9]+)",filename).group(2)
            yield {key:re.split(r"\r\n+",data)}


