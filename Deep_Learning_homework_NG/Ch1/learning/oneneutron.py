#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-24 上午10:11
@Author  : fay
@Email   : fay625@sina.cn
@File    : oneneutron.py
@Software: PyCharm
"""
import numpy as np
import utils

learn_rate = 0.01


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(train_x, train_y, loops=100):
    """
    A["w"]-->|变量w| D
    B["b"] -->|变量b| D
    C["x"] -->|训练集输入| D
    D["z = w.T * x + b"] --> E["a = σ(z)"]
    E --> F["L(a,y) = -yloga - (1-y)log(1-a)"]


    :param train_x:
    :param train_y:
    :param loops:
    :return:
    """
    input_dim = train_x.shape[0]
    m = train_x.shape[1]
    # 初始化w,b
    w = np.zeros(shape=(input_dim, 1))
    b = 0

    for i in range(loops):
        # 前向计算损失
        z = np.dot(w.T, train_x) + b
        A = sigmoid(z)
        cost = -1 / m * np.sum(
            train_y * np.log(A) + (1 - train_y) * np.log(1 - A))

        # 向后更新参数
        dZ = A - train_y
        dw = np.dot(train_x, dZ.T) / m
        db = np.sum(dZ) / m
        w -= learn_rate * dw
        b -= learn_rate * db
        print("After loop", i, "get cost", cost)
    return w, b


def predict(w, b, x):
    num = x.shape[1]
    y_pred = np.zeros(shape=(1, num))
    z = np.dot(w.T, x) + b
    A = sigmoid(z)
    for i in range():
        y_pred[i] = 1 if A[0, i] > 0.5 else 0

    return y_pred


train_x, train_y, test_x, test_y = utils.load_data()
w, b = train(train_x, train_y)
print(predict(w, b, test_x))
