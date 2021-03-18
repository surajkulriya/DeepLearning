#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:16:44 2021

@author: surajkulriya
"""
import numpy as np
import functionFiles
import matplotlib.pyplot as plt
import sigNeuron

class noHiddenLayer:
    inNeurons = 1;
    outNeurons = 1
    w = []
    beta = 1
    accuracy = []
    def __init__(self,inputNeurons, outputClasses,Beta):
        self.inNeurons = inputNeurons
        self.outNeurons = outputClasses
        self.beta = Beta
        # temp = []
        # accuracy = []
        # for i in range(self.inNeurons): temp.append([float(i)])
        for i in range(self.outNeurons): self.w.append([])
        for i in range(self.outNeurons):
            for j in range(self.inNeurons):
                # print(i,j,i*self.outNeurons+j)
                (self.w)[i].append(i*self.outNeurons+j)
    
    def classAcuracy(self, y_true, y_exp):
        ans = 0;
        for i in range(len(y_true)): 
            if(y_true[i]==y_exp[i]): ans+=1
        return ans/len(y_true)
    
    def output(self, x):
        ans = []
        for i in range(self.outNeurons):
            ans.append(np.matmul(x, self.w[i]))
        return ans
    
    def actFunc(self,x):
        return functionFiles.sigmoidActfunc(self.beta, x)
    
    def diffActFunc(self, x):
        return functionFiles.diffSigmoidActfunc(self.beta, x)
    
    def train(self, x, y, y_int, learning_rate, momentum):
        max_epoch = 1000
        for epoch in range(max_epoch):
            y_arr = []
            for i in range(len(x)):
                y_exp = self.output(x[i])
                y_arr.append(y_exp.index(max(y_exp))+1)
                s = []
                for ii in range(len(y_exp)): s.append(self.actFunc(y_exp[ii]))
                for k in range(len(s)):    # k'th output neuron
                    mul = (learning_rate)*(y[i][k]-s[k])*(self.diffActFunc(y_exp[k]))
                    for j in range(len(self.w[k])):  #weight associated with j'th neuron of 1st layer and k'th neuron of 2nd layer
                        self.w[k][j]+= mul*x[i][j]
        
            (self.accuracy).append(self.classAcuracy(y_int, y_arr))
        # print(y_arr)
    
    def classAcuracyVsepoch(self):
        plt.plot(self.accuracy)



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
        


if(__name__=="__main__"):
    # print("suraj")
    
    x = []
    i=0
    y = []
    y_int = []
    
    path ="/home/surajkulriya/DeepLearning/assignment1/data/Classification/LS_Group23/Class1.txt"
    takeInputfromTxtfile(path, x, y, 1, y_int)

    path = "/home/surajkulriya/DeepLearning/assignment1/data/Classification/LS_Group23/Class2.txt"
    takeInputfromTxtfile(path, x, y, 2, y_int)
    
    path = "/home/surajkulriya/DeepLearning/assignment1/data/Classification/LS_Group23/Class3.txt"
    takeInputfromTxtfile(path, x, y, 3, y_int)
    
    p = noHiddenLayer(3, 3, 1)
    # print(getattr(p,"w"))
    p.train(x, y, y_int, 1, 0)
        
    
    
    
    
    
    
    
    
    
    
    
    