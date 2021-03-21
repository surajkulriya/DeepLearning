#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 18:23:26 2021

@author: surajkulriya
"""


import classificationClasses
from sklearn.model_selection import train_test_split
oneHiddenLayer = classificationClasses.oneHiddenLayer
noHiddenLayer = classificationClasses.noHiddenLayer
twoHiddenLayers = classificationClasses.twoHiddenLayers
def takeInputfromTxtfile(path, x, y, Class, y_int):
    file = open(path,"r")
    data = file.read()
    # print(data)
    i = 0
    while(i<(len(data))):
        temp = []
        tempy = [0, 0, 0]
        tempy[Class-1] = 1
        s = ""
        while(data[i]!=' '): 
            s+=data[i]
            i+=1
        i+=1
        temp.append(float(s))
        s=""
        while(data[i]!='\n'):
            s+=data[i]
            i+=1
        temp.append(float(s))
        i+=1
        temp.append(1)
        x.append(temp)
        y.append(tempy)
        y_int.append(Class)
        

def getY(y_int):
    y = []
    for i in range(len(y_int)):
        y.append([0, 0, 0])
        y[i][y_int[i]-1] = 1
    return y


if(__name__=="__main__"):
     
    x = []
    i=0
    y = []
    y_int = []
    
    path ="/home/surajkulriya/DeepLearning/assignment1/data/Classification/NLS_Group23 (copy).txt"
    takeInputfromTxtfile(path, x, y, 1, y_int)
    y_int = []
    for i in range(300): y_int.append(1)
    for i in range(500): y_int.append(2)
    for i in range(1000): y_int.append(3)
    y = getY(y_int)

    
    X_train, X_test, y_train_int, y_test_int = train_test_split(x, y_int, test_size=0.2)
    X_train, X_valid, y_train_int, y_valid_int = train_test_split(X_train, y_train_int, test_size=0.25)
    y_train = getY(y_train_int)
    y_test = getY(y_test_int)
    y_valid = getY(y_valid_int)
    p = twoHiddenLayers(3, 3, 1, 5, 4)
    # print(getattr(p,"w"))
    p.train(x, y, y_int, 1.0, 0, X_test, y_test_int, X_valid, y_valid_int, 500, "non-Linearly")
        
    # p2 = noHiddenLayer(3, 3, 1)
    # p2.train(x, y, y_int, 1.0, 0, X_test, y_test_int, X_valid, y_valid_int, "non-Linearly")    
    
       