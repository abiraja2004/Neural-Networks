
"""
@author: Lebron.Ran
@file: vanillaNN.py
@time: 2017/2/26 0026-21:22

"""

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import matplotlib.pyplot as plt

###   this is a simple implmention of the most vanilla Neural Network with only 1 hidden-layers

#generate a dataset and visualize it

np.random.seed(0)
X,y = sklearn.datasets.make_moons(200,noise=0.20)

# plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)
# plt.show()

num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2

epsilon = 0.01 #learning rate
reg_lambda = 0.01 #regularization parameter

# helper function to plot a decision boundary X: your dataset ; pred_func: your classifier
def plot_decision_boundary(pred_func):

    # the range of padding
    x_min,x_max = X[:,0].min() - 0.5 , X[:,0].max() + 0.5
    y_min,y_max = X[:,1].min() - 0.5 , X[:,1].max() + 0.5

    h = 0.01

    # generate a grid of points with distance h between them.

    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    Z = pred_func(np.c_[xx.ravel(),yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Spectral)

#helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2']

    #forward propagation to calculate the loss  given the parameters

    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)

    #calculate the final loss
    final_log_probs = -np.log(probs[range(num_examples),y])

    data_loss = np.sum(final_log_probs)

    #add regulatization terms

    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return 1.0 / num_examples * data_loss


#the helper function to predict the class label: (0 or 1 here)
def predict(model,x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # forward propagation

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs,axis=1)


# this function learns the parameters for the 3-layer neural networks and return the model
def build_model(nn_hidendim,num_iter = 5000,print_loss=True):

    # initialize the parameters randomly
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim,nn_hidendim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1,nn_hidendim))
    W2 = np.random.randn(nn_hidendim, nn_output_dim) / np.sqrt(nn_hidendim)
    b2 = np.zeros((1,nn_output_dim))

    # the model we will return

    model = {}

    #gradient for the whole batch

    for i in xrange(0,num_iter):
        # forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # backpropagation step
        delta3  = probs
        delta3[range(num_examples),y] -= 1

        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3,axis=0,keepdims=True)

        delta2 = delta3.dot(W2.T) * (1 - np.power(a1,2))

        dW1 = np.dot(X.T,delta2)
        db1  =np.sum(delta2,axis=0)

        #add regulatization terms to W1,W2
        dW2 += reg_lambda * W2;
        dW1 += reg_lambda * W1;

        #update the parameters

        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        #assign the parameters to the latest madel

        model = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}

        if print_loss and (i % 1000 == 0):
            print "the latest loss after the iteration %i:%f" % (i,calculate_loss(model))

    return model

#build a model with 3-dim hidden layer
#
# model = build_model(3,num_iter=20000,print_loss=True)
#
# plot_decision_boundary(lambda x:predict(model,x))
#
# plt.show()

plt.figure(figsize=(16,32))
hidden_layer_dimensions = [1,2,3,4,5,20,50]

for i,nn_num in enumerate(hidden_layer_dimensions):
    plt.subplot(4,2,i+1)
    plt.title("Hidden Layer Size %i" % nn_num)
    model = build_model(nn_num,num_iter=20000,print_loss=False)
    plot_decision_boundary(lambda x:predict(model,x))

plt.show()