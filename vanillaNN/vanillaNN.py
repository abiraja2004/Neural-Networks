
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
def plot_decision_boundary(X,y,pred_func):

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
    final_log_probs = -np.log()

    data_loss = np.sum(final_log_probs)

    #add regulatization terms

    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return 1.0 / num_examples * data_loss


def predict(model,x):
    pass