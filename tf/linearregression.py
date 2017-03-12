#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:47:40 2017

@author: chosenone
"""

import numpy as np
import tensorflow as tf


#build our model
W = tf.Variable([-0.1],tf.float32)
b = tf.Variable([-1.],tf.float32)

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# predictions
y_hat = W * X + b

#calculate the loss give W,b

square_deltas = tf.square(y_hat - y)
loss = tf.reduce_sum(square_deltas)


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


X_train = [1.0,2.0,3.,4.]
y_train = [0.,-1.,-2.,-3.]

for i in range(1000):
    sess.run(train,{X:X_train,y:y_train})
    

# evaluate training accuracy    
print(sess.run([y_hat,loss],{X:X_train,y:y_train}))    

cur_W,cur_b,cur_loss = sess.run([W,b,loss],{X:X_train,y:y_train})

print "W: %s, b: %s, loss: %s." % (cur_W,cur_b,cur_loss)


#==============================================================================
# linear model with tf.contrib.learn, a high-level API 
#==============================================================================

print '################Linear Model With tf.contrib.learn########################'

# declare the features

features = [tf.contrib.layers.real_valued_column('x',dimension=1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1.,-2.,-3.])


input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x},y,batch_size=4,num_epochs=1000)

estimator.fit(input_fn=input_fn,steps=1000)

# in real project,we'd better use a separate validation and testing set to avoid overfitting
estimator.evaluate(input_fn=input_fn)



#==============================================================================
# custom model with tf.contrib.learn, a high-level API 
#==============================================================================

print '#############Custom Model###############'

def model(features,labels,mode):
    
    # build a linear model and predict values
    
    W = tf.get_variable('W',[1],dtype=tf.float64)
    b = tf.get_variable('b',[1],dtype=tf.float64)
    
    y = W * features['x'] + b
    
    # sub-graph of loss

    loss = tf.reduce_sum(tf.square(y - labels))
    
    # sub-graph of training
    
    globle_step = tf.train.get_global_step()
    
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    
    train = tf.group(optimizer.minimize(loss),tf.assign_add(globle_step,1))
    
    return tf.contrib.learn.ModelFnOps(mode=mode,predictions=y,loss=loss,train_op=train)


estimator_custom = tf.contrib.learn.Estimator(model_fn=model)

# dataset and input_fn is the same as memtioned above.

# train 

estimator_custom.fit(input_fn=input_fn,steps=1000)

# evaluste the custom model

estimator_custom.evaluate(input_fn=input_fn,steps=10)




    
    
                   




































    


