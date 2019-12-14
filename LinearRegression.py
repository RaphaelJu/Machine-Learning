#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:23:58 2019

@author: raphaleju
Linear Regression algorithm example, including ridge regression
LinearRegression.py TRAIN TEST RIDGE


"""

import numpy as np
import os
from numpy import linalg as LA
import sys


def QR(D):
    
    """
    QR Decomposition with the Gram-Schmidt Procedure
    """
    d = D.shape[1]
    n = D.shape[0]
    Q = np.ones((n,d))   # orthogonal basis
    R = np.zeros((d,d))  # set of projections
    for i in range(d):   # change R to a I matrix
        R[i][i] = 1   
        
    
    
    for i in range(d):
        Ui = np.copy(D[:,i])
        for j in range(0,i):
            # Update R in each column
            Aj = Q[:,j]
            Pji = (np.dot(Aj.T, Ui)/ LA.norm(Aj)**2)
            R[j,i] = Pji
            Ui -= Pji * Q[:,j]
        
        Q[:,i] = Ui
    
    return(Q,R)
        






def Rw(Q,R,y):
    """
    Calculate Rw = B
    """
    
    d = Q.shape[1]-1
    delta_inv = np.zeros((d+1,d+1))
    delta_d = delta_inv.shape[0]
    
    for i in range(delta_d):
        delta_inv[i,i] = 1/(LA.norm(Q[:,i])**2)
    B = (delta_inv @ (Q.T)) @ y
    return(B)
    





def back_sub(A,b):
    """
    a function use back substraction method to solve Ax = b
    The input A should be a n*n matrix, input B should be a n*1 vector
    the return vector x should be n*1 vector
    """

    
    n = A.shape[1]
    x = np.zeros((n,1))
    for i in range(n):
        index = (n-1)-i
        x_index = b[index]
        for j in range(0,i):
            k = n-1-j 
            x_index -= A[index,k]*b[k]
        x[index] = x_index
    return(x)
    
    
#        
#if __name__ == "__main__":
#    
#    train = np.loadtxt("train.txt",delimiter=",")
#    test = np.loadtxt("test.txt",delimiter=",")
#    
#    
#    train_x = train[:,:-1]
#    train_y = train[:,-1]
#    alpha = 0
#    
#    n = train_x.shape[0]
#    A0 = np.ones((n,1))
#    D_ag = np.hstack((A0,train_x)) # argumented data set
#    
#    A_root = alpha**0.5 * np.eye((D_ag.shape[1]))
#    ## generate D'
#    
#    D = D_ag
#    #D = np.vstack((D_ag,A_root))
#    D_original = D.copy()    
#   
#    y_0 = np.zeros((A_root.shape[0],1))
#    train_y = train_y.reshape((-1,1))
#    y = train_y
#    #y = np.vstack((train_y,y_0))
#    
#    Q, R = QR_factor(D)
#    
#    B = Rw(Q,R,y)   
#
#    
#    w = back_sub(R,B)
#    L2 = (LA.norm(w))**2
#    
#    train_y = train_y.reshape((-1,1))
#    y_predict = D_original @ w
#    y_diff = y_predict - train_y
#    train_y_diff =  train_y - train_y.mean()
#    
#    train_SSE = np.dot(y_diff.T,y_diff)
#    train_TSS = np.dot(train_y_diff.T, train_y_diff)
#    
#    train_R_squared = (train_TSS-train_SSE)/train_TSS


if __name__ == "__main__":

    # load data
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    alpha = float(sys.argv[3])
    
    
    train = np.loadtxt(train_file,delimiter=",")
    test = np.loadtxt(test_file,delimiter=",")
    
    
    train_x = train[:,:-1]
    train_y = train[:,-1]
    
    n = train_x.shape[0]
    A0 = np.ones((n,1))
    D_ag = np.hstack((A0,train_x)) # argumented data set
    
    A_root = alpha**0.5 * np.eye((D_ag.shape[1]))
    
    ## generate D'
    D_original = np.vstack((D_ag,A_root))
    D = D_original.copy()    
    
    
    y_0 = np.zeros((A_root.shape[0],1))
    train_y = train_y.reshape((-1,1))

    y = np.vstack((train_y,y_0))
    
    # Conduct QR factorization and substraction to solve w
    Q, R =  QR(D)
    B = Rw(Q,R,y)   
    w = back_sub(R,B)
    L2 = (LA.norm(w))
    
    
    ## evaluate train dataset
    train_y = train_y.reshape((-1,1))
    train_y_predict = D_ag @ w
    train_y_err = train_y_predict - train_y
    train_y_diff =  train_y - train_y.mean()
    
    train_SSE = np.dot(train_y_err.T,train_y_err)
    train_TSS = np.dot(train_y_diff.T, train_y_diff)
    train_R_squared = (train_TSS-train_SSE)/train_TSS
    
    
    
    ## evaluate test dataset
    test_x = test[:,:-1]
    test_y = test[:,-1]
    
    testD_ag = np.hstack((A0,test_x))
    test_y = test_y.reshape((-1,1))
    test_y_predict = testD_ag @w
    test_y_err = test_y_predict - test_y
    test_y_diff = test_y - test_y.mean()
    
    test_SSE = np.dot(test_y_err.T, test_y_err)
    test_TSS = np.dot(test_y_diff.T, test_y_diff)
    test_R_squared = (test_TSS - test_SSE)/test_TSS
    
    
    print("w:\n",w,"\n")
    print("L2 norm:",L2,"\n")
    print("Train SSE:",train_SSE,"\n")
    print("Train R-squared:", train_R_squared,"\n")
    print("Test SSE:",test_SSE,"\n")
    print("Test R-squared:", test_R_squared,"\n")

### test case, need to be delete when submit the homework
##w1 = LA.solve(R,B)
#
#
#from sklearn.linear_model import LinearRegression
#X = train_x
#y = train_y
##from sklearn.linear_model import Ridge
##clf = Ridge(alpha=1)
##clf.fit(X, y) 
##clf.coef_
##clf.intercept_
#
##reg.score(X,y)
#reg.coef_
#reg.intercept_


#from sklearn.metrics import mean_squared_error
#mean_squared_error(train_y,y_predict)




