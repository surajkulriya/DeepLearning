#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 00:27:05 2021

@author: surajkulriya
"""


import numpy as np
import functionFiles  #self defined library
import matplotlib.pyplot as plt
import sigNeuron     #self defined library
from sklearn.model_selection import train_test_split
import random 
class noHiddenLayer:
    inNeurons = 1;
    outNeurons = 1
    w = []
    beta = 1
    rms_arr = []
    def __init__(self, inputNeurons, outputClasses, Beta):
        self.inNeurons = inputNeurons
        self.outNeurons = outputClasses
        self.beta = Beta
        for i in range(self.outNeurons): self.w.append([])
        for i in range(self.outNeurons):
            for j in range(self.inNeurons):
                # print(i,j,i*self.outNeurons+j)
                (self.w)[i].append(float(random.uniform(-1, 1)))
    
    def rms(self, y_true, y_exp):
        ans = float(0);
        for i in range(len(y_true)): 
            for j in range(len(y_true[i])):
                ans+=(y_true[i][j]-y_exp[i][j])*(y_true[i][j]-y_exp[i][j])
        ans = ans/len(y_true)
        ans=ans**(0.5)
        return ans
    
    def output(self, x):
        ans = []
        for i in range(self.outNeurons):
            ans.append(np.matmul(x, self.w[i]))
        return ans
    
    def actFunc(self,x):
        return functionFiles.LinearFunc(self.beta, x)
    
    def diffActFunc(self, x):
        return functionFiles.diffLinearFunc(self.beta, x)
    
    def getValues(self, x):
        ans = []
        for i in range(len(x)):
            ans.append(self.output(x[i]))
        return ans
    
    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, max_epoch, data_type):
        test_rms = []
        valid_rms= []
        for epoch in range(max_epoch):
            y_arr = []
            for i in range(len(x)):
                y_exp = self.output(x[i])
                y_arr.append(y_exp)
                s = []
                for ii in range(len(y_exp)): s.append(self.actFunc(y_exp[ii]))
                for k in range(len(s)):    # k'th output neuron
                    mul = (learning_rate)*(y[i][k]-s[k])*(self.diffActFunc(y_exp[k]))
                    for j in range(len(self.w[k])):  #weight associated with j'th neuron of 1st layer and k'th neuron of 2nd layer
                        self.w[k][j]+= mul*x[i][j]
        
            (self.rms_arr).append(self.rms(y_int, y_arr))
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            test_rms.append(self.rms(y_test, y_test_exp))
            valid_rms.append(self.rms(y_valid, y_valid_exp))
        # print(y_arr)
        plt.plot(self.rms_arr, label = "train rms")
        plt.plot(test_rms, label = "test rms")
        plt.plot(valid_rms, label = "valid rms")
        plt.title("graph for "+data_type+" regression on no hidden layers")
        plt.legend()
        plt.show()

class oneHiddenLayer:
    inNeuron = 1
    outNeurons = 1
    nHL1 = 1
    beta = 1
    w = [[],[]]
    rms_arr = []
    #in inNeurons, we already have included bias term, and the input x too is supposed to have a extra "1" as bias term
    # but in nHL1, we have not included bias term and thus we are giving nHL1+1 weights to each of output neurons
    def __init__(self, inNeurons, outneurons, Beta, nHiddenLayer1):
        self.inNeuron = inNeurons
        self.outNeurons = outneurons
        self.beta = Beta
        self.nHL1 = nHiddenLayer1
        for i in range(nHiddenLayer1): self.w[0].append([])
        for i in range(nHiddenLayer1):
            for j in range(inNeurons):
                self.w[0][i].append(float(random.uniform(0, 1)))
        for i in range(outneurons): self.w[1].append([])
        for i in range(outneurons):
            for j in range(nHiddenLayer1+1):
                self.w[1][i].append(float(random.uniform(0, 1)))
    
    def rms(self, y_true, y_exp):
        ans = float(0);
        for i in range(len(y_true)): 
            for j in range(len(y_true[i])):
                ans+=(y_true[j]-y_exp[j])*(y_true[j]-y_exp[j])
        ans = ans/len(y_true)
        ans = ans**(0.5)
        return ans
    
    def actFunc(self,x):     #activation function defined for hidden layer 1
        return functionFiles.sigmoidActfunc(self.beta, x)
    
    def diffActFunc(self, x):#diffrentiation of activation function defined for hidden layer 1
        return functionFiles.diffSigmoidActfunc(self.beta, x)
    
    def outActFunc(self,x):  #activation function defined for output layer
        return functionFiles.LinearFunc(self.beta, x)
    
    def outDiffActFunc(self, x): #differentiation of activation function defined for output layer
        return functionFiles.diffLinearFunc(self.beta, x)
    
    def outputDL1(self, x):  # activation values output of of hidden layer 1
        outDLlayer1 = []   #output of Hidden Layer 1
        for i in range(self.nHL1):
            outDLlayer1.append(np.matmul(x, self.w[0][i]))
        return outDLlayer1
    
    def sOut(self, x): # final output of hidden layer 1
        ans = []
        for i in range(len(x)):
            ans.append(self.actFunc(x[i]))
        return ans
    
    def factOut(self, outDLlayer1): #activation value output of output layer
        ans = []
        for i in range(self.outNeurons):
            ans.append(np.matmul(outDLlayer1, self.w[1][i]))
        return ans
    
    def fout(self, x): # final output of output layer
        ans = []
        for i in range(len(x)):
            ans.append(self.outActFunc(x[i]))
        return ans 
    
    def getValues(self, x):
        ans = []
        for i in range(len(x)):
            a1 = self.outputDL1(x[i])
            s1 = self.sOut(a1)
            s1.append(float(0))
            a2 = self.factOut(s1)
            s2 = self.fout(a2)
            ans.append(s2)
        return ans
            
    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, max_epoch, data_type):
        test_rms= []
        valid_rms= []
        for epoch in range(max_epoch):
            y_arr = []
            for tuplex in range(len(x)):
                a1 = self.outputDL1(x[tuplex])
                s1 = self.sOut(a1)
                s1.append(float(1.0)) #bias term
                a2 = self.factOut(s1)
                s2 = self.fout(a2)
                y_arr.append(s2)
                for outN in range((self.outNeurons)):
                    for HL in range(self.nHL1):
                        mul = (learning_rate)*float(self.w[1][outN][HL])*float(y[tuplex][outN]-s2[outN])*float(self.outDiffActFunc(a2[outN]))*float(self.diffActFunc(a1[HL]))
                        for iN in range(self.inNeuron):
                            # print(self.w[0][HL][iN][0])
                            # print(self.w[0][HL][iN],mul*x[tuplex][iN])
                            self.w[0][HL][iN]+= mul*x[tuplex][iN]        
                    mul = (learning_rate)*(y[tuplex][outN]-s2[outN])*self.outDiffActFunc(a2[outN])
                    for HL in range((self.nHL1+1)):
                        self.w[1][outN][HL]+= mul*s1[HL]
            self.rms_arr.append(self.rms(y_int, y_arr))
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            test_rms.append(self.rms(y_test, y_test_exp))
            valid_rms.append(self.rms(y_valid, y_valid_exp))
        # print(y_arr)
        plt.plot(self.rms_arr, label = "train rms")
        plt.plot(test_rms, label = "test rms")
        plt.plot(valid_rms, label = "valid rms")
        plt.legend()
        plt.title("graph for "+data_type+" regression on one hidden layer")
        plt.show()
    
class twoHiddenLayers:
    inNeuron = 1
    outNeurons = 1
    nHL1 = 1
    nHL2 = 1
    beta = 1
    w = [[],[], []]
    rms_arr = []
    #in inNeurons, we already have included bias term, and the input x too is supposed to have a extra "1" as bias term
    # but in nHL1, we have not included bias term and thus we are giving nHL1+1 weights to each of output neurons
    def __init__(self, inNeurons, outneurons, Beta, nHiddenLayer1, nHiddenLayer2):
        self.inNeuron = inNeurons
        self.outNeurons = outneurons
        self.beta = Beta
        self.nHL1 = nHiddenLayer1
        self.nHL2 = nHiddenLayer2
        for i in range(nHiddenLayer1): self.w[0].append([])
        for i in range(nHiddenLayer1):
            for j in range(inNeurons):
                self.w[0][i].append(float(random.uniform(0, 1)))
        for i in range(nHiddenLayer2): self.w[1].append([])
        for i in range(nHiddenLayer2):
            for j in range(nHiddenLayer1+1):
                self.w[1][i].append(float(random.uniform(0, 1)))
        for i in range(outneurons): self.w[2].append([])
        for i in range(outneurons):
            for j in range(nHiddenLayer2+1):
                self.w[2][i].append(float(random.uniform(0, 1)))
    
    def rms(self, y_true, y_exp):
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
            a1 = self.outputDL1(x[i])
            s1 = self.sOut(a1)
            s1.append(float(1.0))
            a2 = self.outputDL2(s1)
            s2 = self.sOut(a2)
            s2.append(float(1.0))
            a3 = self.factOut(s2)
            s3 = self.fout(a3)
            y_arr.append(s3)
        return y_arr

    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, max_epoch, data_type):
        test_rms= []
        valid_rms= []
        for epoch in range(max_epoch):
            y_arr = []
            for tuplex in range(len(x)):
                a1 = self.outputDL1(x[tuplex])
                s1 = self.sOut(a1)
                s1.append(float(1.0)) #bias term
                a2 = self.outputDL2(s1)
                s2 = self.sOut(a2)
                s2.append(float(1.0)) #bias term
                a3 = self.factOut(s2)
                s3 = self.fout(a3)
                y_arr.append(s3)
                for outN in range((self.outNeurons)):
                    for HL1 in range(self.nHL1):
                        dsum = float(0)
                        for HL2 in range(self.nHL2):
                            dsum+=self.w[2][outN][HL2]*self.diffActFunc(a2[HL2])*self.w[1][HL2][HL1]
                        mul = (learning_rate)*(y[tuplex][outN]-s3[outN])*(self.outDiffActFunc(a3[outN]))*self.diffActFunc(a1[HL1])
                        for iN in range(self.inNeuron):
                            self.w[0][HL1][iN]+=mul*x[tuplex][iN]*dsum
                    for HL2 in range(self.nHL2):
                        mul = (learning_rate)*(y[tuplex][outN]-s3[outN])*self.outDiffActFunc(a3[outN])*self.w[2][outN][HL2]*self.diffActFunc(a2[HL2])
                        for HL1 in range(self.nHL1+1):
                            self.w[1][HL2][HL1]+=mul*s1[HL1]
                    mul = (learning_rate)*(y[tuplex][outN]-s3[outN])*(self.outDiffActFunc(a3[outN]))
                    for HL2 in range(self.nHL2+1):
                        self.w[2][outN][HL2]+=mul*(s2[HL2])
            self.rms_arr.append(self.rms(y_int, y_arr))
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            test_rms.append(self.rms(y_test, y_test_exp))
            valid_rms.append(self.rms(y_valid, y_valid_exp))
        # print(y_arr)
        plt.plot(self.rms_arr, label = "train accuracy")
        plt.plot(test_rms, label = "test accuracy")
        plt.plot(valid_rms, label = "valid accuracy")
        plt.legend()
        plt.title("graph for "+data_type+" regression on two hidden layers")
        plt.show()

