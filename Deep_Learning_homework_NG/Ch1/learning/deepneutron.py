#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-24 下午2:32
@Author  : fay
@Email   : fay625@sina.cn
@File    : deepneutron.py
@Software: PyCharm
"""
import numpy as np
import utils

learn_rate = 0.0075
loops = 3000


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    A = np.maximum(0, Z)
    return A


def init_parameters(layer_dims):
    np.random.seed(1)
    params = {}
    layers = len(layer_dims)

    for l in range(1, layers):
        w = np.random.randn(layer_dims[l], layer_dims[l - 1])
        w = w * 0.01 / np.sqrt(layer_dims[l - 1])
        params['W' + str(l)] = w
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return params


def update_parameters(parameters, grads, learn_rate):
    layers = len(parameters) // 2
    for l in range(layers):
        parameters["W" + str(l + 1)] -= learn_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learn_rate * grads["db" + str(l + 1)]

    return parameters


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T)) / m
    cost = np.squeeze(cost)
    return cost


def relu_activation_forward(A_prev, W, b):
    Z = W.dot(A_prev) + b
    A = relu(Z)
    return A, (A_prev, W, b), Z


def sigmoid_activation_forward(A_prev, W, b):
    Z = W.dot(A_prev) + b
    A = sigmoid(Z)
    return A, (A_prev, W, b), Z


def deep_model_forward(X, parameters):
    caches = []
    A = X
    layers = len(parameters) // 2

    for layer in range(1, layers):
        A_prev = A
        A, awb, z = relu_activation_forward(A_prev,
                                            parameters['W' + str(layer)],
                                            parameters['b' + str(layer)])
        caches.append((awb, z))

    AL, awb, z = sigmoid_activation_forward(A, parameters['W' + str(layers)],
                                            parameters['b' + str(layers)])
    caches.append((awb, z))

    return AL, caches


def linear_backward(dZ, awb):
    A_prev, W, b = awb
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def relu_activation_backward(dA, cache):
    awb, Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    dA_prev, dW, db = linear_backward(dZ, awb)
    return dA_prev, dW, db


def sigmoid_activation_backward(dA, cache):
    awb, Z = cache
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    dA_prev, dW, db = linear_backward(dZ, awb)
    return dA_prev, dW, db


def deep_model_backward(AL, Y, caches):
    grads = {}
    layers = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[layers - 1]
    current_layer = str(layers)
    dA_prev_temp, dW_temp, db_temp = sigmoid_activation_backward(dAL,
                                                                 current_cache)
    grads["dA" + current_layer] = dA_prev_temp
    grads["dW" + current_layer] = dW_temp
    grads["db" + current_layer] = db_temp

    for layer in reversed(range(layers - 1)):
        current_cache = caches[layer]
        dA_prev_temp, dW_temp, db_temp = relu_activation_backward(
            grads["dA" + str(layer + 2)], current_cache)
        grads["dA" + str(layer + 1)] = dA_prev_temp
        grads["dW" + str(layer + 1)] = dW_temp
        grads["db" + str(layer + 1)] = db_temp

    return grads


def deep_layer_model(X, Y, layers_dims):
    parameters = init_parameters(layers_dims)

    for i in range(0, loops):
        AL, caches = deep_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = deep_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learn_rate)
        print("Cost after loops %i: %f" % (i, cost))

    return parameters


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = deep_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("Accuracy: " + str(np.sum((p == y) * 1. / m)))
    return p


train_x, train_y, test_x, test_y = utils.load_data()

input_dim = 4
hidden_units = 7
output_dim = 1
layer_dims = (input_dim, hidden_units, output_dim)

model = deep_layer_model(train_x, train_y, layer_dims)
predict(test_x, test_y, model)
