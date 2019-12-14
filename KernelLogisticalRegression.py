#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:20:32 2019

@author: raphaleju
Kernel Logistic Regression 
Run as following:
KernelLogisticalRegression.py TRAIN TEST [linear | quadratic | gaussian] [spread]
"""

import numpy as np
import os
import sys
from numpy import linalg as LA



def linear_kernel(D,Z):
    return(D@Z.T)

def quadratic_kernel(D,Z):
    XZ = D@Z.T+1
    return(np.multiply(XZ,XZ))

def gaussian_kernel(D,Z,spread):
    K_X2 = np.multiply(D,D).sum(axis=1).reshape(-1,1)
    K_Z2 = np.multiply(Z,Z).sum(axis=1).reshape(1,-1)
   
    norm_squared_matrix = K_X2 + K_Z2
    XZ = D@Z.T
    power = -(norm_squared_matrix - 2*XZ)/(2*spread**2)
    K = np.exp(1)**(power)
    return K



def classify(Y_vec):
    filter_1 = Y_vec >= 0.5
    Y_vec[:] = 0
    Y_vec[filter_1] = 1
    return Y_vec

def cal_acc(y_pred,y_true):
    count = sum(y_pred == y_true)
    return(count/len(y_true))
    
if __name__ == "__main__":
    
    # load data
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    kernel = sys.argv[3]
    if len(sys.argv)==5:
        spread = float(sys.argv[4])
    
    Train = np.loadtxt(train_file, delimiter=",")
    Test = np.loadtxt(test_file, delimiter=",")
    
    alpha = 0.01

    
    
    D = Train[:,:-1]
    DZ = Test[:,:-1]
    Y_train = Train[:,-1]
    Y_test = Test[:,-1]
    #K = kernel_matrix()
    
    # Calculate K in different kernel
    if kernel == "linear":
        K_XX = linear_kernel(D,D)
        K_XZ = linear_kernel(D,DZ)
    elif kernel == "quadratic":
        K_XX = quadratic_kernel(D,D)
        K_XZ = quadratic_kernel(D,DZ)
    elif kernel == "gaussian":
        K_XX = gaussian_kernel(D,D,spread)
        K_XZ = gaussian_kernel(D,DZ,spread)
    
    K_hat = K_XX + 1
    mixture_inv = LA.inv(K_hat + alpha*np.identity(K_hat.shape[0]))
    c = (mixture_inv @ Y_train).reshape(-1,1)
    Y_train_pred_num = K_hat @ c
    Y_train_pred = classify(Y_train_pred_num.copy())

    Y_train = Y_train.reshape(-1,1)
    train_acc = cal_acc(Y_train_pred,Y_train)
    
    KZ_hat = K_XZ + 1
    Y_test_pred_num = (c.T @ KZ_hat).reshape(-1,1)
    Y_test_pred = classify(Y_test_pred_num.copy())
    
    Y_test = Y_test.reshape(-1,1)
    test_acc = cal_acc(Y_test_pred,Y_test)
    print("The accuracy of test dataset for",kernel, "kernel:",test_acc)
    
    
    
    
    
    
    