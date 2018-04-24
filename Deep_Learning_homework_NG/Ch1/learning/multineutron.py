#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-24 下午1:29
@Author  : fay
@Email   : fay625@sina.cn
@File    : multineutron.py
@Software: PyCharm
"""
import numpy as np
import utils

hidden_units = 4
lr = 1.2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(X, Y, loops=1000):
    m = X.shape[1]
    input_dim = X.shape[0]
    output_dim = Y.shape[0]
    W1 = np.random.randn(hidden_units, input_dim)
    b1 = np.zeros(shape=(hidden_units, 1))
    W2 = np.random.randn(output_dim, hidden_units)
    b2 = np.zeros(shape=(output_dim, 1))

    for i in range(loops):
        # forward propagate
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        loss = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
        loss = - np.sum(loss) / m
        print("After loops ", i, " get loss ", loss)
        # backward propagate
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # update params
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def predict(X,Y,params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    y_pred = np.round(A2)

    accuracy = (np.dot(Y, y_pred.T) + np.dot(1 - Y, 1 - y_pred.T)) / float(Y.size) * 100
    return y_pred, accuracy


train_x, train_y, test_x, test_y = utils.load_data()
params = train(train_x, train_y, loops=1000)

_, accuracy = predict(test_x, test_y, params)
print("Accuracy is", accuracy)
