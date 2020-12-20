#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:03:53 2020

@author: zoe
"""

from autograd import numpy as np
from autograd import grad, elementwise_grad
from autograd.misc.optimizers import adam, sgd
from sklearn.datasets import make_moons
import numpy.random as npr
import pandas as pd
import numpy 
import scipy as sp
import matplotlib.pyplot as plt


def myentropy(nn_model, weightlist, xdata, returnallp=False):
    '''
    Usage: for NN_Dropout, use the same weights, duplicated N times
    for MFVI, pass the sampled weights
    '''

    #assert xdata.shape[0]==2
    n_samples = xdata.shape[1]
    p1narray = np.zeros((len(weightlist), n_samples)) #NWeightSamples x NPoints
   
    if type(nn_model) != list: 
        for i, w in enumerate(weightlist):
            w = np.reshape(w, (1, nn_model.D))
            p1narray[i, :] = nn_model.forward(w, xdata) #assumes that the 'model.forward' is dropout-like and has generates different outputs for each i
    elif type(nn_model) == list: # deterministic 
        for i, nn in enumerate(nn_model): 
            p1narray[i, :] = nn.forward(weightlist[i], xdata)
    #print (p_here.shape)
# <<<<<<< HEAD
    certainpts = np.logical_or(np.all(p1narray==0, axis=0), np.all(p1narray==1, axis=0)) 

    p2narray = 1 - p1narray
    p1narraym = np.mean(p1narray, axis=0)
    p2narraym = np.mean(p2narray, axis=0)
    Hpredcheck = -p1narraym*np.log(p1narraym) - p2narraym*np.log(p2narraym)
    Hpredcheck[certainpts] = 0.0
    if returnallp:
        return p1narray, p1narraym, Hpredcheck
    else:
        return p1narraym, Hpredcheck


# calculate the accuracy for MC dropout
def auc_calc(x_test, y_test, nn, N, perc, model, weightlist=None): 
    ''' 
    Options for model "mc", "bbvi", "ensemble", "deterministic"
    For BBVI, pass a list of weights. 
    '''
    p = []
    n_test = len(y_test)
    if model != "deterministic": 
        if model == "mc":
            p_mean, entropymean = myentropy(nn, [nn.weights]*N, x_test.T)
        elif (model == "bbvi") and weightlist is not None: 
            p_mean, entropymean = myentropy(nn, weightlist, x_test.T)
        elif model == "ensemble": # deterministic 
            nn_weights = [] 
            for nn_here in nn: 
                nn_weights.append(nn_here.weights) 
            p_mean, entropymean = myentropy(nn, nn_weights, x_test.T)

        idx = np.argsort(entropymean)
        y_test = y_test[idx]
        p_mean = p_mean[idx]
        y_pred_retained = p_mean[0:int(perc*n_test)] # choosing samples with smallest entropy to evaluate 
        y_test_retained = y_test[0:int(perc*n_test)]
        predict_proba = np.round(y_pred_retained)
  
        auc = len(y_test_retained[predict_proba==y_test_retained]) / len(y_pred_retained) * 100
    else: 
        auc = auc_calc_proba(x_test, y_test, nn, N, perc)
        
    return auc


# calculate the accuracy for MC dropout
def auc_calc_beta(x_test, y_test, nn, N, perc, model, weightlist=None): 
    ''' 
    Options for model "mc", "bbvi", "ensemble", "deterministic"
    For BBVI, pass a list of weights. 
    '''
    p = []
    n_test = len(y_test)
    if model != "deterministic": 
        if model == "mc":
            p_allw, p_mean, entropymean = myentropy(nn, [nn.weights]*N, x_test.T, returnallp=True)
        elif (model == "bbvi") and weightlist is not None: 
            p_allw, p_mean, entropymean = myentropy(nn, weightlist, x_test.T, returnallp=True)
        elif model == "ensemble": # deterministic 
            p_allw, p_mean, entropymean = myentropy(nn, weightlist, x_test.T, returnallp=True)
        #p_allw has dimension: NWeightSamples x NXData
        idx = np.argsort(entropymean)
        y_test = y_test[idx]
        p_mean = p_mean[idx]
        p_allw = p_allw[:, idx]
        y_pred_retained_allw = p_allw[:, 0:int(perc*n_test)]
        y_pred_retained = p_mean[0:int(perc*n_test)] # choosing samples with smallest entropy to evaluate 
        y_test_retained = y_test[0:int(perc*n_test)]
        ypredmean = np.round(y_pred_retained)
        ypred_allw = np.round(y_pred_retained_allw) #NW x NX
        auc_allw = np.zeros(ypred_allw.shape[0])
        for w in range(ypred_allw.shape[0]):
            auc_allw[w] = np.count_nonzero(ypred_allw[w, :]==y_test_retained)/len(y_test_retained) * 100
        return auc_allw
    else: 
        auc = auc_calc_proba(x_test, y_test, nn, N, perc)
        return auc #this only returns the mean accuracy

#>>>>>>> a08c25b69ac8768b2b32d2fa3d5e240076410cbb

# calculate the accuracy for deterministic model
def auc_calc_proba(x_test, y_test, nn, N, perc):
    n_samples = len(y_test)
    auc = np.zeros(N)
    for j in range(N):
        p_here = nn.forward(nn.weights, x_test.T)[0][0]
        idx = np.argsort(p_here)
        i2 = int((1-perc/2)*n_samples)
        i1 = int(perc*n_samples/2)
        idx1 = idx[0:i1] # indices predicted to be 0 
        idx2 = idx[i2:] # indices predicted to be 1 
        x0 = x_test[idx1]
        y0 = y_test[idx1]
        y1 = y_test[idx2]
        # print(len(y1[y1==1])+len(y0[y0==0]), len(y0)+len(y1), (len(y1[y1==1])+len(y0[y0==0]))/(n_samples*perc))
        auc[j] = (len(y0[y0==0]) + len(y1[y1==1]))/(len(y0) + len(y1)) * 100
    return auc

#<<<<<<< HEAD

# NEED TO PLOT THE TRAINING DATA SEPARATELY!
def plot_entropycontours(x, y, model, weightlist, ax, title, poly_degree=1, test_points=None, shaded=True, interval=np.arange(-6, 6, 0.1)):
    '''
    plot_decision_boundary plots the training data and the decision boundary of the classifier.
    input:
       x - a numpy array of size N x 2, each row is a patient, each column is a biomarker
       y - a numpy array of length N, each entry is either 0 (no cancer) or 1 (cancerous)
       models - an array of classification models
       ax - axis to plot on
       poly_degree - the degree of polynomial features used to fit the model
       test_points - test data
       shaded - whether or not the two sides of the decision boundary are shaded
    returns: 
       ax - the axis with the scatter plot
    
    '''
    # Plot data
    # ax.scatter(x[y == 1, 0], x[y == 1, 1], alpha=0.2, c='red', label='class 1')
    # ax.scatter(x[y == 0, 0], x[y == 0, 1], alpha=0.2, c='blue', label='class 0')
    
    # Create mesh
    #interval = np.arange(-6, 6, 0.1)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    xx = np.concatenate((x1, x2), axis=1)
    
    yy = myentropy(model, weightlist, xx.T)[1]  
    yy = yy.reshape((n, n))
    
    cl = ax.imshow(yy, origin='lower', extent=(min(interval), max(interval), min(interval), max(interval)))
    
    ax.set_title(title)
    if test_points is not None:
        for i in range(len(test_points)):
            pt = test_points[i]
            if i == 0:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black', label='test data')
            else:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black')


    lowb = interval[0]+0.5
    highb = interval[-1]-0.5    
    ax.set_xlim((lowb, highb))
    ax.set_ylim((lowb, highb))
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.legend(loc='best')
    return cl



if __name__ == '__main__':
    pass