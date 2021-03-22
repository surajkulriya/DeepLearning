#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:32:14 2021

@author: surajkulriya
"""

# this solution is for univariate data

import numpy as np 
import pandas as pd
from perceptronFile import perceptron          #self defined library 
from sklearn.model_selection import train_test_split
import regressionClasses
noHiddenLayer = regressionClasses.noHiddenLayer
oneHiddenLayer = regressionClasses.oneHiddenLayer
twoHiddenLayers = regressionClasses.twoHiddenLayers

def splitXY(d):
    X=[]
    y=[]
    cols = d.columns
    for i in range(len(cols)-1):
        X.append(list(data.loc[:,cols[i]]))
    temp = []
    for i in range(len(X[0])): temp.append(1)
    X.append(temp)
    X=np.transpose(X)
    Y = []
    # X=list(X)
    # for i in range(len(X)): X[i].append(1)
    y = list(data.loc[:,cols[-1]])
    Y.append(y)
    Y = np.transpose(Y)
    return(X,Y)

if (__name__=="__main__"):
    
    data = pd.read_csv("/home/surajkulriya/Downloads/Group23/Regression/UnivariateData/data.csv")
    X, Y= splitXY(data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)
    
    p0 = noHiddenLayer(2, 1, 1)
    p0.train(X_train, y_train, y_train, 0.01, 0, X_test, y_test, X_valid, y_valid, 100, "univariate")
    
    p1 = oneHiddenLayer(2, 1, 1, 5)
    p1.train(X_train, y_train, y_train, 0.1, 0, X_test, y_test, X_valid, y_valid, 100, "univariate")
    
    p2 = twoHiddenLayers(2, 1, 1, 5, 4)
    p2.train(X_train, y_train, y_train, 1, 0, X_test, y_test, X_valid, y_valid, 30, "univariate")
    
    
    
    
    
    
    
    
    
    
    