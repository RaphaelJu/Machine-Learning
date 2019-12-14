#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:31:45 2019

@author: raphaleju
Expectation-Maximization Clustering
Run as followings:
EMClustering.py FILE k eps
file is the clustering filename
k is number of clusters
eps is error tolerance

"""

import numpy as np
import os
from numpy import linalg as LA
import sys

def prob_gaussian(x,avg,cov,cov_inv,determ):
    #fix = np.exp(-(x-avg).T@cov_inv@(x-avg)/2) * 1/((np.pi)**(d/2)*LA.det(cov))
    fix = np.exp(-(x-avg).T@cov_inv@(x-avg)/2) * 1/((np.pi)**(d/2))*determ
    return fix
    
    
    

def post_prob(X,mean_vec,covs,determs,cov_invs,prior_vec,i_index,j_index,prob_fun):
    cov = covs[i]
    cov_inv = cov_invs[i]
    avg = mean_vec[:,i].reshape(-1,1)
    prior_prob = prior_vec[0][i]
    x = X[j,:].reshape(-1,1)
    determ = determs[i]
    numer = prob_gaussian(x,avg,cov,cov_inv,determ)*prior_prob
    denom = 0
    for l in range(len(covs)):
        cov = covs[l]
        cov_inv = cov_invs[l]
        avg = mean_vec[:,l].reshape(-1,1)
        denom += prob_gaussian(x,avg,cov,cov_inv,determ)*prior_vec[0][l]
    wij = numer/denom
    return wij
    
    
    
    
    

if __name__ == "__main__":
    #X = np.loadtxt('iris.txt', delimiter=',', usecols=[0,1,2,3])
    file = sys.argv[1]
    k = int(sys.argv[2])
    eps = float(sys.argv[3])
    
#    file = "iris.txt"
#    k = 3
#    eps = 0.1
    
    
    # hard coding to read iris dataset, if all variables are numerical, then usual way to load data
    # will be fine
    try:
        data = np.loadtxt(file,delimiter = ',')
        X = data[:,:-1]
        Y = data[:,-1].reshape(-1,1)
    except:
        X = np.loadtxt(file, delimiter=',', usecols=[0,1,2,3])
        Y = np.loadtxt(file, delimiter=',', usecols=[4],dtype=np.str)
        t = 1
        for category in np.unique(Y):
            Y[Y==category] = t
            t += 1
        Y = Y.reshape(-1,1).astype(int)
            
    

    #m,n = np.unique(Y,return_counts= True)


    
    n = X.shape[0]
    d = X.shape[1]
    mean_vector = np.zeros((d,k))
    # calculate the min and max value for each column in dataset
    dimension_max = np.amax(X,axis=0)
    dimension_min = np.amin(X,axis=0)
    # initialize mean_vectors randomly, following uniformal distribution between min and max value
    for i in range(d):
        for j in range(k):
            mean_vector[i,j] = np.random.uniform(dimension_min[i],dimension_max[i])
    # initialize k covariance matrix and store them in a list called cov_list
    cov_list = []
    for i in range(k):
        cov_list.append(np.eye(d))
    # initialize prior probabilities and store them into a vector 
    prior_vector = np.ones((1,k))/k
    # here is the same because inverse of an identity matrix is an identity matrix
    cov_inv_list = cov_list.copy()
    # initialize determinant matrix
    determs = []
    for i in range(k):
        determs.append(LA.det(cov_list[i]))
    
    t = 0
    
    # The reason that cov_inv and determinant list is pre-calculated is the cost of these two calculation 
    # is computationally high, thus we calculate them and store them in variables first
    # when updates covariance matrix, we will do the same 
    
    lamda = 0.0001
    
    while True:
        t += 1
        mean_vector_old = mean_vector.copy()
        # Expectation Step
        # n is the row number of the dataset and k the number of clusters
        # W is weight matrix 
        W = np.zeros((n,k))
        for i in range(k):
            for j in range(n):
                #weight_matrix[i,j] = post_prob(X,mean_vector,cov_list,prior_vector,i,j,prob_gaussian)
                W[j,i] = post_prob(X,mean_vector,cov_list,determs,cov_inv_list,prior_vector,i,j,prob_gaussian)

        # Maximization Step
        cov_list = []
        cov_inv_list = []
        determs = []
        Z = X.copy()
        for i in range(k):
            # re-estimate mean
            mean_vector[:,i] = X.T@W[:,i]/np.sum(W[:,i])
            # re-estimate covariance matrix
            Zi = X - 1*mean_vector_old[:,i].reshape(-1,1).T
            numer = np.zeros((d,d))
            for j in range(n):
                numer += W[j,i]*(Zi[j,:].reshape(-1,1)@Zi[j,:].reshape(-1,1).T)
            denom = np.sum(W[:,i])
            cov = numer/denom + lamda*np.eye(d)
            cov_inv = LA.inv(cov)
            determ = LA.det(cov)
            cov_list.append(cov)
            cov_inv_list.append(cov_inv)
            determs.append(determ)
            # re-estimate priors
            prior_vector[0][i] = np.sum(W[:,i])/n
        diff_mean = mean_vector - mean_vector_old
        #print(LA.norm(diff_mean))
        if LA.norm(diff_mean) < eps:
            break
        
    Y_hat = Y -1
    # extract clustering result and calculate purity score
    cluster_indexs = {}
    clustering = np.argmax(W,axis=1).reshape(-1,1)
    purity_num = 0
    for i in range(k):
        num_match_i = 0
        i_index = np.argwhere(clustering == i)[:,0]
        for j in range(k):
            j_index = np.argwhere(Y_hat == j)[:,0]
            intersect = np.intersect1d(i_index,j_index)
            insect_num = len(intersect)
            if insect_num >= num_match_i:
                num_match_i = insect_num
                cluster_indexs[j] = i_index
        purity_num += num_match_i
    for i in range(k):
        if i not in cluster_indexs.keys():
            cluster_indexs[i] = []
            
    purity_score = purity_num/n
    print("Purity Score:",purity_score)
    
    print("Mean for each cluster:")
    print(mean_vector)        
    for i in range(k):
        print("Covariance for cluster"+str(i+1))
        print(cov_list[i])
    print("Number of iterations:",t)
    for i in cluster_indexs.keys():
        print("The row that classified as cluster"+str(i+1))
        print(cluster_indexs[i])
    for i in range(k):
        print("Size for cluster"+str(i+1)+": "+str(len(cluster_indexs[i])))
        
        
            

        
    
    