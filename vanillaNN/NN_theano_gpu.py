#coding=utf8
"""
@author: Lebron.Ran
@file: NN_theano_gpu.py
@time: 2017/3/1 0001-10:08
"""
# 学习theano 在开启GPU加速的情况下 的使用，以及比较加速效果
import theano
import theano.tensor as T
import sklearn.datasets
import datetime
import numpy as np

# default float data type
theano.config.floatX = 'float32'

np.random.seed(10)
X_train,y_train = sklearn.datasets.make_moons(5000,noise=0.20)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32)
y_train_onehot = np.eye(2)[y_train]


# print X_train.shape

# some size definitions
m = len(y_train)
nn_hidden_dim = 10
nn_input_dim = 2
nn_output_dim = 2


epsilon = np.float32(0.015) # learning rate, to keep the result of calculation stays in 'float32'
reg_lambda = np.float32(0.01) # regularization term .same as the above


# convert the training set and parameters to float32 to save them on GPU!!!!

X = theano.shared(X_train.astype('float32'))
y = theano.shared(y_train_onehot.astype('float32'))

W1 = theano.shared(np.random.randn(nn_input_dim, nn_hidden_dim).astype('float32'),name='W1')
b1 = theano.shared(np.zeros(nn_hidden_dim).astype('float32'),name='b1')
W2 = theano.shared(np.random.randn(nn_hidden_dim, nn_output_dim).astype('float32'),name='W2')
b2 = theano.shared(np.zeros(nn_output_dim).astype('float32'),name='b2')

# define the forward propagation
z1 = X.dot(W1) + b1
s1 = T.tanh(z1)
z2 = s1.dot(W2) + b2
y_hat = T.nnet.softmax(z2)

# the regularization term
loss_reg = (1.0/ m) * (reg_lambda / 2) * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)))
# loss function
loss = T.nnet.binary_crossentropy(y_hat,y).mean() + loss_reg

# make the predictions
prediction = T.argmax(y_hat,axis=1)

#automatic gradient
dW1 = T.grad(loss,W1)
db1 = T.grad(loss,b1)
dW2 = T.grad(loss,W2)
db2 = T.grad(loss,b2)


forward_prop = theano.function([],y_hat)
calculate_loss = theano.function([],loss)
predict = theano.function([],prediction)

gradient_step = theano.function([],updates=( (W1,W1 - epsilon * dW1),
                                             (W2,W2 - epsilon * dW2),
                                             (b1,b1 - epsilon * db1),
                                             (b2,b2 - epsilon * db2)))

def build_model(num_passes=20000,print_loss=False):

    np.random.seed(0)
    W1.set_value((np.random.randn(nn_input_dim,nn_hidden_dim) / np.sqrt(nn_input_dim)).astype('float32'))
    b1.set_value(np.zeros(nn_hidden_dim).astype('float32'))
    W2.set_value((np.random.randn(nn_hidden_dim, nn_output_dim) / np.sqrt(nn_hidden_dim)).astype('float32'))
    b2.set_value(np.zeros(nn_output_dim).astype('float32'))

    # gradient descent
    for i in np.arange(0,num_passes):

        gradient_step()

        if print_loss and i % 1000 == 0:
            print 'Loss after iteration %d : %f .' % (i,calculate_loss())


#testing
begin = datetime.datetime.now()

gradient_step()

end = datetime.datetime.now()

print 'time of one gradient step: %s' % (end - begin)

begin = datetime.datetime.now()

build_model(print_loss=True)

end = datetime.datetime.now()

print 'time of build model: %s' % (end - begin)