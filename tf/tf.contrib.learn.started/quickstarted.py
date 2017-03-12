#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:28:19 2017

@author: chosenone
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

tf.logging.set_verbosity(tf.logging.INFO)

# Datasets

IRIS_TRAIN_SET = 'iris_training.csv'
IRIS_TEST_SET = 'iris_test.csv'

# training set and test set
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
           IRIS_TRAIN_SET,target_dtype=np.int,features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
           IRIS_TEST_SET,target_dtype=np.int,features_dtype=np.float32)


#specify all features have real-value 
feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]

# define a 3-hidden-layers DNN classifiers with 10,20,10 respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir='/tmp/iris_model',
                                            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=0.1)
                                            )


# monitor api

validation_metrics = {
        'accuracy':tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_accuracy,
                                                           prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
        'precision':tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_precision,
                                                           prediction_key="classes"),
        'recall':tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_recall,
                                                           prediction_key="classes")                                                    
        }

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                                    test_set.data,
                                    test_set.target,
                                    every_n_steps=50,
                                    metrics=validation_metrics,
                                    early_stopping_metric="loss",
                                    early_stopping_metric_minimize=True,
                                    early_stopping_rounds=50
                                    )

# fit model
classifier.fit(training_set.data,training_set.target,steps=2000,monitors=[validation_monitor])




# evaluate the accuracy

accuracy_score = classifier.evaluate(x=test_set.data,y=test_set.target)['accuracy']


print('Accuracy: %f' % accuracy_score)

# classify two new samples

new_samples = np.array([[6.4,3.2,4.5,1.5],
                        [5.8,3.1,5.0,1.7]],dtype=float)

y_hat = list(classifier.predict(new_samples,as_iterable=True))

print("Predictions: %s" % str(y_hat))
































print(np.shape(training_set.data))


