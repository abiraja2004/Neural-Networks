#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:26:20 2017

@author: chosenone
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import time 
import data_util
import datetime

from CNN4Text import CNN4Text
from tensorflow.contrib import learn

import csv

#==============================================================================
# parameters
#==============================================================================
#==============================================================================
# Data parameters
#==============================================================================
tf.flags.DEFINE_string("data_postive_path","./data/rt-polaritydata/rt-polarity.pos"
                       ,"postive data path")
tf.flags.DEFINE_string("data_negative_path","./data/rt-polaritydata/rt-polarity.neg"
                       ,"negative data path")

#==============================================================================
# evaluate parameters
#==============================================================================

tf.flags.DEFINE_integer("batch_size",64,"Batch size(default 64)")
tf.flags.DEFINE_string("checkpoint_path","","checkpoint path from training runs")
tf.flags.DEFINE_boolean("Eval_Training_Set",False,"Evaluate on the whole traing set?")

#==============================================================================
# Misc parameters
#==============================================================================

tf.flags.DEFINE_bool("allow_soft_parameters",True,"allow soft device placement(default true)")
tf.flags.DEFINE_bool("log_device_placement",False,"Log placement of ops on devices(default false)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")

for attr,value in sorted(FLAGS.__flags.items()):
    print("%s:%s" % (attr.upper(),value))

print("\n")

# Load data





















































