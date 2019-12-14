#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:43:03 2019

@author: raphaleju
Neural Networks 

Run as the following
NeuralNetworks.py TRAIN TEST m η epochs
m is the hidden layer
η is the learning step size
epochs is the number of runs
"""


import numpy as np
import os
import sys
from numpy import linalg as LA



def reLU(neth):
    """
    ReLU activation function
    The input is an column vector, neth
    The return value is a column vector zi
    """
    filter_0 = (neth < 0)
    neth[filter_0] = 0
    return neth

def softmax(neto):
    """
    Softmax activation function 
    The input is an column vector, neto
    The return value is a column vector oi
    """
    numerator = np.exp(neto)
    denominator = sum(numerator)
    oi = numerator/denominator
    return oi


def MLP_SGD(train,m, eta, epochs):
    
    train_Y =  train[:,-1].astype(int)
    train_y_classes = int(max(train_Y))
    train_hot_index = train_Y - 1
    train_y = np.eye(train_y_classes)[train_hot_index]
    train_x = train[:,:-1]
    

    d = train_x.shape[1]
    p = train_y_classes
    
    bh = np.random.uniform(-0.1,0.1,(m,1))
    bo = np.random.uniform(-0.1,0.1,(p,1))
    
    wh = np.random.uniform(-0.1,0.1,(d,m))
    wo = np.random.uniform(-0.1,0.1,(m,p))
    
    t = 0
    n = train_x.shape[0]
    
    while True:
        index_new = np.arange(n)
        np.random.shuffle(index_new)
        D_random = train_x[index_new,]
        y_random = train_y[index_new,]
        for i in range(n):
            xi = D_random[i,].reshape(-1,1)
            yi = y_random[i,].reshape(-1,1)
            # feed-forward phase
            neth = bh + np.dot(wh.T,xi)
            zi = reLU(neth)
            neto = bo + np.dot(wo.T, zi)
            oi = softmax(neto)
#            print("bh",bh)
#            print("xi",xi)
#            print("wh.T@xi",np.dot(wh.T,xi))
#            print("neth",neth)
#
#            print("neto",neto)
#            print("wo",wo)
#            print("i",i)
            
            
            # Backpropagation phase: net gradients
            delta_o = oi-yi
            deriv = np.zeros_like(zi)
            deriv[zi>0] = 1
            delta_h = np.multiply(deriv,(np.dot(wo,delta_o)))
            # gradient descent for bias vector
            grad_bo = delta_o
            grad_bh = delta_h
            bo = bo - eta*grad_bo
            bh = bh - eta*grad_bh
            # gradient descent for weight matrices
            grad_wo = np.dot(zi, delta_o.T)
            grad_wh = np.dot(xi, delta_h.T)
            wo = wo - eta*grad_wo
            wh = wh - eta*grad_wh
          
        t += 1
        if t > epochs:
            return (wh,wo,bh,bo)

def pred_y(D, model_result):  
    wh = model_result[0]
    wo = model_result[1]
    bh = model_result[2]
    bo = model_result[3]
    netH = bh + np.dot(wh.T,D.T)
    Z = reLU(netH)
    netO = bo + np.dot(wo.T,Z)
    O = softmax(netO)
    return O.T
    

def accuracy(pred_y,y): 
    pred_y_index = np.argmax(pred_y, axis=1)
    y_index = np.argmax(y,axis=1)
    correct = sum(pred_y_index == y_index)
    return correct/len(pred_y_index)
    
if __name__ == "__main__":
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    m = float(sys.argv[3])
    eta = float(sys.argv[4])
    epochs = float(sys.argv[5])
    m = int(m)
#    train_file = "shuttle.trn.txt"
#    test_file = "shuttle.tst.txt"
#    m = 5
#    eta = 0.0001
#    epochs = 1
    
    train = np.loadtxt(train_file,delimiter = ",")
    test  = np.loadtxt(test_file,delimiter = ",")
    
    


    wh, wo, bh, bo = MLP_SGD(train,m, eta, epochs)
    MLP_SGD_model = [wh, wo, bh, bo]

    train_Y =  train[:,-1].astype(int)
    train_y_classes = int(max(train_Y))
    train_hot_index = train_Y - 1
    train_y = np.eye(train_y_classes)[train_hot_index]
    train_x = train[:,:-1]
    
    
    test_Y  =  test[:,-1].astype(int)
    test_y_classes  = int(max(test_Y))
    test_hot_index = test_Y - 1
    test_y  = np.eye(test_y_classes)[test_hot_index]
    test_x  = test[:,:-1]
    
    train_pred_y = pred_y(train_x, MLP_SGD_model)
    test_pred_y = pred_y(test_x, MLP_SGD_model)
    test_accuracy = accuracy(test_pred_y,test_y)
    train_accuracy = accuracy(train_pred_y,train_y)
    

    print()
    print("Train Accuracy", train_accuracy)
    print("Test Accuracy", test_accuracy)
    print("Weight of hidden layer:\n",wh)
    print("Weight of ouput layer:\n",wo)
    print("Bias of hidden layer:\n",bh)
    print("Bias of output layer:\n",bo)
    
        
    

