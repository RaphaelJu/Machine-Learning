#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:02:29 2019

@author: raphaleju

SVM based on SGA
Run as the followings:
SVM.py TRAIN TEST C eps [linear OR quadratic OR gaussian ] spread    

"""

import numpy as np
import os

from numpy import linalg as LA
import sys

def linear_kernel(X,Z):
    return X@Z.T

def quadratic_kernel(X,Z):
    XZ = X@Z.T
    return np.multiply(XZ,XZ)
    
def gaussian_kernel(X,Z,spread):
    K_X2 = np.multiply(X,X).sum(axis=1).reshape(-1,1)
    K_Z2 = np.multiply(Z,Z).sum(axis=1).reshape(1,-1)
   
    norm_squared_matrix = K_X2 + K_Z2
    XZ = X@Z.T
    power = -(norm_squared_matrix - 2*XZ)/(2*spread**2)
    K = np.exp(1)**(power)
    return K

def sign(hx):
    result = np.zeros_like(hx)
    result[hx>0] = 1
    result[hx<0] = -1
    return result

def accuracy(predy,y):
    acc = sum(predy==y)/len(y)
    return acc


def SGA(train_Y,alpha_new,K_XXhat,steps):
    t = 0
    train_n = train_Y.shape[0]
    while True:
        alpha = alpha_new.copy()
        for k in range(train_n):
            delta = np.sum(np.multiply( np.multiply(alpha_new,train_Y),K_XXhat[:,k].reshape(-1,1)))

            gradient = (1-train_Y[k]*delta)
            alpha_k = alpha[k] + steps[k]*gradient
            if alpha_k < 0:
                alpha_k = 0
            if alpha_k > C:
                alpha_k = C
            alpha_new[k] = alpha_k
        
        if LA.norm(alpha-alpha_new) <= eps:
            return alpha_new
        t = t+1



if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    C = float(sys.argv[3])
    eps = float(sys.argv[4])
    kernel = sys.argv[5]
    
    if kernel == 'gaussian':
        spread = float(sys.argv[6])
    
    train = np.loadtxt(train_file,delimiter = ',')
    test = np.loadtxt(test_file, delimiter = ',')
    
#    train_file = "train.txt"
#    test_file = "test.txt"
#    train = np.loadtxt(train_file,delimiter = ',')
#    test = np.loadtxt(test_file, delimiter = ',')
#    C = 1
#    kernel = input("Choose a kernel: ")
#    spread = 1
#    eps = float(input("Choose your tolerance: "))
    
    train_X = train[:,:-1]
    train_Y = train[:,-1].reshape(-1,1)
    test_X  = test[:,:-1]
    test_Y  = test[:,-1].reshape(-1,1) 
    

    
    if kernel == "linear":
        K_XX = linear_kernel(train_X,train_X)
        K_XZ = linear_kernel(train_X,test_X)
    elif kernel == "quadratic":
        K_XX = quadratic_kernel(train_X,train_X)
        K_XZ = quadratic_kernel(train_X,test_X)
    elif kernel == "gaussian":
        K_XX = gaussian_kernel(train_X,train_X,spread)
        K_XZ = gaussian_kernel(train_X,test_X,spread)
    K_XXhat = K_XX + 1
    K_XZhat = K_XZ + 1
    
    
    
    steps = 1/np.diag(K_XXhat).reshape(-1,1)
    alpha_new = np.zeros_like(steps)
    
    # SGA
    
    
    alpha = SGA(train_Y,alpha_new,K_XXhat,steps)
    
    SV_index = np.where(alpha >0)[0]
    SV_alpha = alpha[SV_index,:]
    SV_Y = train_Y[SV_index,:]

    train_n = train_X.shape[0]
    # Use KXX matrix to predict train y
    train_hx = np.zeros_like(train_Y)
    for i in range(train_n):
        K_train = K_XXhat[SV_index,i].reshape(-1,1)
        train_hx[i] = sum(np.multiply(np.multiply(SV_alpha,SV_Y),K_train))
    train_pred_Y = sign(train_hx)   
    train_accuracy = accuracy(train_pred_Y, train_Y)
    
    # Use KXZ matrix to predict test y 
    test_n = test.shape[0]
    test_hx = np.zeros_like(test_Y)
    for i in range(test_n):
        K_test = K_XZhat[SV_index,i].reshape(-1,1)
        test_hx[i] = sum(np.multiply(np.multiply(SV_alpha,SV_Y),K_test))
    test_pred_Y = sign(test_hx)
    test_accuracy = accuracy(test_pred_Y,test_Y)
    # Support Vectors
    SVs = train_X[SV_index,:]
    # Support Vectors with index at the first column
    SupportVectors = np.hstack((SV_index.reshape(-1,1),SVs))
    

    for i in range(len(SupportVectors)):
        print("X"+str(int(SupportVectors[i,0]))+": "+str(SupportVectors[i,1:]))
    
    print()
    print("Alpha:\n", SV_alpha)

    if kernel == 'linear':
        phiX = np.hstack((train_X,np.ones_like(train_Y).reshape(-1,1)))
    if kernel == 'quadratic':
        d = SVs.shape[1]
        phid = int(d + d*(d-1)/2 + 1)
        #nSVs = SVs.shape[0]
        phiX = np.ones((train_n,phid))
        train_X_squared = np.multiply(train_X,train_X)
        col_index = d  # column index for (2**0.5)*XiXj
        for i in range(d):
            phiX[:,i] = train_X_squared[:,i]
            for j in range(i+1,d):
                phiX[:,col_index] = 2**0.5*np.multiply(train_X[:,i],train_X[:,j])
                col_index += 1
                
                
    if kernel == 'linear'or kernel == 'quadratic':           
                
        w_matrix = np.multiply(np.multiply(SV_alpha,SV_Y),phiX[SV_index,:])
        w = w_matrix.sum(axis=0).reshape(-1,1)
        print()
        print("Weight for",kernel,"Kernel:")
        print(w)
#        hx_phi = (w.T@phiX.T).reshape(-1,1)
#        pred_phi_y = sign(hx_phi)
#        acc_phi = accuracy(pred_phi_y,train_Y)
#        print(acc_phi,'acc_phi')
#        print()
    print()
    print("Test Accuracy: ",test_accuracy)
#    print("Train Accuracy", train_accuracy)