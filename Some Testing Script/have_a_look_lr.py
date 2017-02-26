#coding=utf8
"""

@author: Lebron.Ran
@file: have_a_look_lr.py
@time: 2017/2/26 0026-19:44


"""
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import matplotlib.pyplot as plt

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


#generate a dataset and visualize it

np.random.seed(0)
X,y = sklearn.datasets.make_moons(200,noise=0.20)
plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X,y)

plot_decision_boundary(X,y,lambda x:clf.predict(x))

plt.title("Logistic Regression")

plt.show()








