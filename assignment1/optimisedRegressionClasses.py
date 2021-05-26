#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:22:44 2021

@author: surajkulriya
"""



import numpy as np
import functionFiles  #self defined library
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sigNeuron     #self defined library
from sklearn.model_selection import train_test_split
import random 



class DLmodel:
    
    inNeuron = 1
    outNeurons = 1
    nHL1 = 0
    nHL2 = 0
    beta = 1
    w=[]
    rms_arr = []
    def rms(self, y_true, y_exp): #rms values obtained for given actual y values and predicted y values
        ans = float(0);
        for i in range(len(y_true)): 
            for j in range(len(y_true[i])):
                ans+=(y_true[j]-y_exp[j])*(y_true[j]-y_exp[j])
        ans = ans/len(y_true)
        ans=ans**(0.5)
        return ans
    

    def actFunc(self,x):     #activation function defined for hidden layers
        return functionFiles.sigmoidActfunc(self.beta, x)
    
    def diffActFunc(self, x):#diffrentiation of activation function defined for hidden layers 
        return functionFiles.diffSigmoidActfunc(self.beta, x)
    
    def outActFunc(self,x):  #activation function defined for output layer
        return functionFiles.LinearFunc(self.beta, x)
    
    def outDiffActFunc(self, x): #differentiation of activation function defined for output layer
        return functionFiles.diffLinearFunc(self.beta, x)
    
    def outputDL1(self, x):  # activation values output of of hidden layer 1 
        outDLlayer1 = []   #output of Hidden Layer
        for i in range(self.nHL1):
            outDLlayer1.append(np.matmul(x, self.w[0][i]))
        return outDLlayer1
    
    def outputDL2(self, x):  # activation values output of of hidden layer 2
        outDLlayer2 = []   #output of Hidden Layer
        for i in range(self.nHL2):
            outDLlayer2.append(np.matmul(x, self.w[1][i]))
        return outDLlayer2
    
    def sOut(self, x): # final output of hidden layers, given input activation values
        ans = []
        for i in range(len(x)):
            ans.append(self.actFunc(x[i]))
        return ans
    
    def factOut(self, outDLlayer2): #activation value output of output layer
        ans = []
        for i in range(self.outNeurons):
            ans.append(np.matmul(outDLlayer2, self.w[2][i]))
        return ans
    
    def fout(self, x): # final output of output layer, given input of activation values
        ans = []
        for i in range(len(x)):
            ans.append(self.outActFunc(x[i]))
        return ans 
    
    def getValues(self, x):
        y_arr = []
        for i in range(len(x)):
            if(nHL1>0):
                a1 = self.outputDL1(x[i])
                s1 = self.sOut(a1)
                s1.append(float(1.0))
            else:
                s1=x
                
            if(nHL2>0):
                a2 = self.outputDL2(s1)
                s2 = self.sOut(a2)
                s2.append(float(1.0))
            else:
                s2=s1
                
            a3 = self.factOut(s2)
            s3 = self.fout(a3)
            y_arr.append(s3)
        return y_arr
    
    def xVSy(self, x_train, y_train, graph_title_add, data_type ):
        if(len(self.w))
        if(len(x_train[0])==2):
            y_train_exp = self.getValues(x_train)
            x_train_x = []
            for i in range(len(x_train)): x_train_x.append(x_train[i][0])
            
            plt.scatter(x_train_x,y_train, label = ("true Y " + data_type))
            plt.scatter(x_train_x,y_train_exp, label = ("model Y "+data_type))
            plt.title(data_type+" data for two hidden layers for univariate data"+graph_title_add)
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
            
        elif(len(x_train[0])==3):
            y_train_exp = self.getValues(x_train)
            x_train_x = []; x_train_y = []
            
            for i in range(len(x_train)): x_train_x.append(x_train[i][0])
            
            for i in range(len(x_train)): x_train_y.append(x_train[i][1])
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_train_x, x_train_y, np.transpose(y_train)[0], label = "Y_"+data_type+" true")
            ax.scatter3D(x_train_x, x_train_y, np.transpose(y_train_exp)[0], label = "Y_"+data_type+" model")
            plt.title("graph for "+data_type+" data for two hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
    
    def modelVSexact(self, x, y, data_type):
        y_exp = self.getValues(x)
        plt.scatter(y, y_exp)
        plt.title("scatter plot for "+ data_type+" on two hidden layers")
        plt.xlabel("Y_true")
        plt.ylabel("Y_model_output")
        plt.show()
        
    def dTypeVSerror(self, x_train, y_train, x_test, y_test, x_valid, y_valid ):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_train_exp = self.getValues(x_train)
        y_train_rms = self.rms(y_train, y_train_exp)
        
        y_test_exp = self.getValues(x_test)
        y_test_rms = self.rms(y_test, y_test_exp)
        
        y_valid_exp = self.getValues(x_valid)
        y_valid_rms = self.rms(y_valid, y_valid_exp)
    
        error = [y_train_rms, y_test_rms, y_valid_rms]
        ax.plot(["Train Data", "Test Data", "Validation Data"], error)
        plt.show()
    
            
    
    
        