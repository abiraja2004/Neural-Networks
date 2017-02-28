#coding=utf8
"""
@author: Lebron.Ran
@file: RNN_numpy.py
@time: 2017/2/28 0028-16:32

"""
import numpy as np

class RNN_numpy:
    def __init__(self,word_dim,hidden_dim=100,bptt_truncate=4):

        # assign variable
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # randomly initial the networks parameters
        self.U = np.random.uniform(-np.sqrt(1.0/word_dim),np.sqrt(1.0/word_dim),(hidden_dim,word_dim))
        self.V = np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(word_dim,hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim))

    if __name__ == '__main__':
        def forward_propagation(self,x):

            # the total number of time steps
            T = len(x)

            # during the process of forward propagation,we need to save the Ss to avoid calculate them later
            # also,we add an additional element for the initial hidden layer ,which we set to 0s.
            s = np.zeros((T+1,self.hidden_dim))
            s[-1] = np.zeros(self.hidden_dim)

            # the output for each time step ,and again we save them for later.
            y = np.zeros((T,self.word_dim))

            #during each time step..
            for t in range(T):

                # this is a simple expression of UX + Ws_t,because X is just a one-hot vector
                s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
                y[t] = np.softmax(self.V.dot(s[t]))

            return [y,s]


