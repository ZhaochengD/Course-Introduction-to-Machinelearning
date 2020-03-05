#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:16:49 2019

@author: zhaocheng_du
"""

import numpy as np
import sys

TRAIN_IN     = sys.argv[1]
TEST_IN      = sys.argv[2]
TRAIN_OUT    = sys.argv[3]
TEST_OUT     = sys.argv[4]
METRIC_OUT   = sys.argv[5]
EPOCH        = int(sys.argv[6])
HIDDEN_UNITS = int(sys.argv[7])
INI_STRATEGY = int(sys.argv[8])
LRATE        = float(sys.argv[9])

def splitData(filename):
    data = np.genfromtxt(filename, delimiter=',').astype(np.int)
    X_star = data[:, 1:]
    Y_star = data[:, 0]
    Y_hot = np.zeros((data.shape[0], 10))
    Y_hot[np.arange(data.shape[0]), Y_star] = 1
    return Y_star, Y_hot, np.hstack((np.ones(X_star.shape[0]).reshape(-1, 1), X_star))

def initParam(X, Y, INI_STRATEGY):
    if INI_STRATEGY == 1:
        alpha_star = np.random.uniform(-0.1, 0.1, (HIDDEN_UNITS, X.shape[1] - 1))
        alpha = np.hstack((np.zeros((HIDDEN_UNITS, 1)), alpha_star))
        beta_star  = np.random.uniform(-0.1, 0.1, (10, HIDDEN_UNITS))
        beta  = np.hstack((np.zeros((10, 1)), beta_star))
    else:
        alpha = np.zeros((HIDDEN_UNITS, X.shape[1]))
        beta  = np.zeros((10, HIDDEN_UNITS + 1))
    return alpha, beta

def NNForward(X_item, Y_item, alpha, beta):
    a = alpha.dot(X_item)
    z_star = 1 / (1 + np.exp(-a))
    z = np.vstack((np.array([1]), z_star))
    b = beta.dot(z)
    Y_hat = np.exp(b) / np.sum(np.exp(b))
    loss = -np.sum(Y_item.T.dot(np.log(Y_hat)))
    obj = {'a': a, 'z_star': z_star, 'z': z, 'b': b, 'Y_hat': Y_hat, 'loss':loss}
    return obj

def NNbackward(X_item, Y_item, alpha, beta, obj):
    g_b = obj['Y_hat'] - Y_item
    g_beta = g_b.dot(obj['z'].T)
    g_z = (g_b.T.dot(beta[:, 1:])).T
    g_a = g_z * (obj['z_star'] * (1 - obj['z_star']))
    g_alpha = g_a.dot(X_item.T)
    return g_alpha, g_beta

def SGD(X_item, Y_item, alpha, beta):
    obj = NNForward(X_item, Y_item, alpha, beta)
    g_alpha, g_beta = NNbackward(X_item, Y_item, alpha, beta, obj)
    alpha -= LRATE * g_alpha
    beta -= LRATE * g_beta
    return alpha, beta

def train(X, Y, alpha, beta):
    for idx in range(X.shape[0]): 
        X_item = X[idx, :].reshape(-1, 1)
        Y_item = Y[idx, :].reshape(-1, 1)
        alpha, beta = SGD(X_item, Y_item, alpha, beta)
    return alpha, beta

def predict(X, Y, alpha, beta):
    result = []
    for idx in range(X.shape[0]):  
        X_item = X[idx, :].reshape(-1, 1)
        Y_item = Y[idx, :].reshape(-1, 1)
        obj = NNForward(X_item, Y_item, alpha, beta)
        result.append(np.argmax(obj['Y_hat']))
    return result

def loss(X, Y, alpha, beta):
    loss  = []
    for idx in range(X.shape[0]):  
        X_item = X[idx, :].reshape(-1, 1)
        Y_item = Y[idx, :].reshape(-1, 1)
        obj = NNForward(X_item, Y_item, alpha, beta)
        loss.append(obj['loss'])
    return np.mean(loss)

def error(result, Y_star):
    err = 0
    for idx in range(len(result)):
        if result[idx] != Y_star[idx]:
            err += 1
    return err / len(result)

def writeLabel(filename, label_li):
    for i in label_li:
        filename.write(str(i) + '\n')

if __name__ == "__main__":
    train_real, trainY, trainX = splitData(TRAIN_IN)
    test_real, testY, testX = splitData(TEST_IN)
    alpha, beta = initParam(trainX, trainY, INI_STRATEGY)
    
    fTrainLab = open(TRAIN_OUT, 'w')
    fTestLab = open(TEST_OUT, 'w')
    fMatric = open(METRIC_OUT, 'w')
    
    print(TRAIN_IN, TEST_IN ,trainX.shape, testX.shape, TRAIN_OUT, METRIC_OUT, EPOCH, HIDDEN_UNITS, INI_STRATEGY, LRATE)
    
    for ep in range(EPOCH):
        alpha, beta = train(trainX, trainY, alpha, beta)
        train_loss = loss(trainX, trainY, alpha, beta)
        test_loss = loss(testX, testY, alpha, beta)
        print(train_loss)
        print(test_loss)
        fMatric.write('epoch={} crossentropy(train): {}'.format(ep + 1, train_loss) + '\n')
        fMatric.write('epoch={} crossentropy(test): {}'.format(ep + 1, test_loss) + '\n')
    
    train_pred = predict(trainX, trainY, alpha, beta)
    test_pred = predict(testX, testY, alpha, beta)
    train_err   = error(train_pred, train_real)
    test_err    = error(test_pred, test_real)
    
    fMatric.write('error(train): {}\n'.format(train_err))
    fMatric.write('error(test): {}\n'.format(test_err))
    
    writeLabel(fTrainLab, train_pred)
    writeLabel(fTestLab, test_pred)