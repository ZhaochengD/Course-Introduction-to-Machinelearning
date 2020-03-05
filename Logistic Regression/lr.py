#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:42:12 2019

@author: zhaocheng_du
"""

import numpy as np
import time
import sys

TRAINI  = sys.argv[1]
VALIDI  = sys.argv[2]
TESTI   = sys.argv[3]
DICT    = sys.argv[4]
TRAINO  = sys.argv[5]
TESTO   = sys.argv[6]
METRICO = sys.argv[7]
EPOCH   = sys.argv[8]

def convertXY(filename):
    f = open(filename)
    X, Y= [],[]
    for line in f:
        Y.append(int(line[0]))
        X_temp = line[2:].strip('\n').split('\t')
        X.append([int(i[:-2]) for i in X_temp])
    return X, Y

def writeLable(predLi, filename):
    of = open(filename, 'w')
    for pred in predLi:
        of.write(str(pred) + '\n')
    of.close()

def writeError(trainError, testError, filename):
    of = open(filename, 'w')
    of.write("error(train): " + str(train_error) + '\n')
    of.write("error(test): " + str(testError))

class LR():
    def __init__(self, X, Y, thetaVec):
        self.X = X
        self.Y = Y
        self.n = len(Y)
        self.thetaVec = thetaVec
        self.k = len(thetaVec)
        self.gradLi    = np.zeros((self.k, 1))
    
    def calThetaTx(self, xi):
        self.thetaTx = 0
        for wordIdx in xi:
            self.thetaTx += self.thetaVec[wordIdx]
        self.thetaTx += self.thetaVec[-1]
        self.expx = np.exp(self.thetaTx)

    def oneGrad(self, i):
        self.gradLi = np.zeros((self.k, 1))
        grad = - self.Y[i] + self.expx / (1 + self.expx)
        for j in self.X[i]:
            self.gradLi[j] = grad
        self.gradLi[-1] = grad
    
    def update(self):
        self.thetaVec -= 0.1 * self.gradLi
    
    def train(self, epoch):
        for ep in range(epoch):
            for i in range(5):
                self.calThetaTx(self.X[i])
                self.oneGrad(i)
                self.update()

                print('theta'+str(i), self.thetaVec)
                print('grad'+str(i), self.gradLi)
        
    def predict(self, X_test):
        pred = []
        for Xi in X_test: 
            self.calThetaTx(Xi)
            prob = 1 / (1 + self.expx)
            pred.append(int(prob < 0.5))
        return pred
    
    def metric(self, Y_real, Y_pred):
        error = 0
        for i in range(len(Y_pred)):
            if Y_real[i] != Y_pred[i]:
                error += 1
        return error / len(Y_pred)

if __name__ == "__main__":
    start = time.time()
    train_X, train_Y = convertXY(TRAINI)
    test_X,  test_Y  = convertXY(TESTI)
    lr = LR(train_X, train_Y, np.zeros((39176, 1)))
    lr.train(int(EPOCH))
    train_pred  = lr.predict(train_X)
    test_pred   = lr.predict(test_X)
    train_error = lr.metric(train_pred, train_Y)
    test_error  = lr.metric(test_pred, test_Y)
    writeLable(train_pred, TRAINO)
    writeLable(test_pred, TESTO)
    writeError(train_error, test_error, METRICO)
    end = time.time()
    print("The time is: ", end - start)
    
