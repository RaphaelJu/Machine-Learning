#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:46:26 2019

@author: raphaleju
Logistic Regression by using SGA method
Run as following:
LogisticRegression.py TRAIN TEST eps eta 
eps is error tolerance and eta is learning rate

"""

import numpy as np
import os
import sys
from numpy import linalg as LA


def sigmoid(x):
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid


def LogitRegresSGA(train, step ,tol):
    """
    This is the function to calculate the weights of variables in Logistic Regression Model 
    by using SGA(Stochastic Gradient Ascent) method
    Input: 
    train is the train dataset, the last column should be response  variable
    step is the step to update the w 
    tol is the stop criterion
    Output:
    A vector, the 
    """
    train_x = train.copy()[:,:-1]
    train_y = train.copy()[:,-1].reshape(-1,1)
    n = train.shape[0]
    D = np.hstack((np.ones((n,1)),train_x)) # get argumented X matrix
    d = D.shape[1]
    


    # GE
    t = 0
    w0 = np.zeros((d,1))

    
    
        
    while True:
        #np.random.seed(t)
        w1 = w0.copy()
        index_new = np.arange(n)
        np.random.shuffle(index_new)
        D_random = D[index_new,]
        y_random = train_y[index_new,]
        for i in range(n):
            xi = D_random[i,].reshape(-1,1)
            gradient = (y_random[i] - sigmoid(np.dot(w0.T,xi)))*xi
            w1 = w1 + step*gradient
        if (LA.norm(w1-w0)<= tol):
            return(w1,t)
        w0 = w1.copy()
        t = t+1
        
        
        

        
        
def Classify(D,w):
    result = sigmoid(D@w)
    for i in range(len(result)):
        if result[i] < 0.5:
            result[i] = 0
        else:
            result[i] = 1
    return result
        

def accuracy(train_predict_y,train_y):
    total = train_y.shape[0]
    correct = sum(train_predict_y == train_y)
    return(correct/total)
    



if __name__ == "__main__":
    
    # load dataset
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    Train = np.loadtxt(train_file,delimiter=",")
    Test = np.loadtxt(test_file,delimiter=",")
    
    eps = float(sys.argv[3])
    eta = float(sys.argv[4])
    
    w,t = LogitRegresSGA(Train, eta ,eps)


    # train accuracy
    train_x = Train.copy()[:,:-1]
    n = train_x.shape[0]
    D = np.hstack((np.ones((n,1)),train_x))
    train_y = Train.copy()[:,-1].reshape(-1,1)
    train_predict_y = Classify(D,w)
    train_acc = accuracy(train_predict_y,train_y)
    
    
    # test accuracy
    
    test_x = Test.copy()[:,:-1]
    n = test_x.shape[0]
    D = np.hstack((np.ones((n,1)),test_x))
    test_y = Test.copy()[:,-1].reshape(-1,1)
    
    test_predict_y = Classify(D,w)
    test_acc = accuracy(test_predict_y,test_y)
    
    print("w:\n",w)
    print("Accuracy for train dataset:",train_acc)
    print("Accuracy for test dataset:",test_acc)