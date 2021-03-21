#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:16:44 2021

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
                (self.w)[i].append(float(random.uniform(0, 1)))
    
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
    
    def getClassLabels(self, x):
        y = []
        for i in range(len(x)):
            y_exp = self.output(x[i])
            y.append(y_exp.index(max(y_exp))+1)
        return y
    
    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, max_epoch, data_type):
        test_accuracy = []
        valid_accuracy = []
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
            y_test_exp = self.getClassLabels(x_test)
            y_valid_exp = self.getClassLabels(x_valid)
            test_accuracy.append(self.classAcuracy(y_test, y_test_exp))
            valid_accuracy.append(self.classAcuracy(y_valid, y_valid_exp))
        # print(y_arr)
        plt.plot(self.accuracy, label = "train accuracy")
        plt.plot(test_accuracy, label = "test accuracy")
        plt.plot(valid_accuracy, label = "valid accuracy")
        plt.title("graph for "+data_type+" seprable classes on no hidden layers")
        plt.legend()
        plt.show()
    
    def classAcuracyVsepoch(self):
        plt.plot(self.accuracy)


class oneHiddenLayer:
    inNeuron = 1
    outNeurons = 1
    nHL1 = 1
    beta = 1
    w = [[],[]]
    accuracy = []
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
    
    def classAcuracy(self, y_true, y_exp):
        ans = 0;
        for i in range(len(y_true)): 
            if(y_true[i]==y_exp[i]): ans+=1
        return ans/len(y_true)
    
    def actFunc(self,x):     #activation function defined for hidden layer 1
        return functionFiles.sigmoidActfunc(self.beta, x)
    
    def diffActFunc(self, x):#diffrentiation of activation function defined for hidden layer 1
        return functionFiles.diffSigmoidActfunc(self.beta, x)
    
    def outActFunc(self,x):  #activation function defined for output layer
        return functionFiles.sigmoidActfunc(self.beta, x)
    
    def outDiffActFunc(self, x): #differentiation of activation function defined for output layer
        return functionFiles.diffSigmoidActfunc(self.beta, x)
    
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
    
    def getClassLabels(self, x):
        y_arr = []
        for i in range(len(x)):
            a1 = self.outputDL1(x[i])
            s1 = self.sOut(a1)
            s1.append(float(1.0))
            a2 = self.factOut(s1)
            s2 = self.fout(a2)
            y_arr.append(a2.index(max(a2))+1)
        return y_arr

    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, max_epoch, data_type):
        test_accuracy = []
        valid_accuracy = []
        for epoch in range(max_epoch):
            y_arr = []
            for tuplex in range(len(x)):
                a1 = self.outputDL1(x[tuplex])
                s1 = self.sOut(a1)
                s1.append(float(1.0)) #bias term
                a2 = self.factOut(s1)
                s2 = self.fout(a2)
                y_arr.append(a2.index(max(a2))+1)
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
            self.accuracy.append(self.classAcuracy(y_int, y_arr))
            y_test_exp = self.getClassLabels(x_test)
            y_valid_exp = self.getClassLabels(x_valid)
            test_accuracy.append(self.classAcuracy(y_test, y_test_exp))
            valid_accuracy.append(self.classAcuracy(y_valid, y_valid_exp))
        # print(y_arr)
        plt.plot(self.accuracy, label = "train accuracy")
        plt.plot(test_accuracy, label = "test accuracy")
        plt.plot(valid_accuracy, label = "valid accuracy")
        plt.legend()
        plt.title("graph for "+data_type+" seprable classes on one hidden layer")
        plt.show()
    
class twoHiddenLayers:
    inNeuron = 1
    outNeurons = 1
    nHL1 = 1
    nHL2 = 1
    beta = 1
    w = [[],[], []]
    accuracy = []
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
    
    def classAcuracy(self, y_true, y_exp):
        ans = 0;
        for i in range(len(y_true)): 
            if(y_true[i]==y_exp[i]): ans+=1
        return ans/len(y_true)
    
    def actFunc(self,x):     #activation function defined for hidden layers
        return functionFiles.sigmoidActfunc(self.beta, x)
    
    def diffActFunc(self, x):#diffrentiation of activation function defined for hidden layers 
        return functionFiles.diffSigmoidActfunc(self.beta, x)
    
    def outActFunc(self,x):  #activation function defined for output layer
        return functionFiles.sigmoidActfunc(self.beta, x)
    
    def outDiffActFunc(self, x): #differentiation of activation function defined for output layer
        return functionFiles.diffSigmoidActfunc(self.beta, x)
    
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
    
    def getClassLabels(self, x):
        y_arr = []
        for i in range(len(x)):
            a1 = self.outputDL1(x[i])
            s1 = self.sOut(a1)
            s1.append(float(1.0))
            a2 = self.outputDL2(s1)
            s2 = self.sOut(a2)
            s2.append(float(1.0))
            a3 = self.factOut(s2)
            y_arr.append(a3.index(max(a3))+1)
        return y_arr

    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, max_epoch, data_type):
        test_accuracy = []
        valid_accuracy = []
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
                y_arr.append(a3.index(max(a3))+1)
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
            self.accuracy.append(self.classAcuracy(y_int, y_arr))
            y_test_exp = self.getClassLabels(x_test)
            y_valid_exp = self.getClassLabels(x_valid)
            test_accuracy.append(self.classAcuracy(y_test, y_test_exp))
            valid_accuracy.append(self.classAcuracy(y_valid, y_valid_exp))
        # print(y_arr)
        plt.plot(self.accuracy, label = "train accuracy")
        plt.plot(test_accuracy, label = "test accuracy")
        plt.plot(valid_accuracy, label = "valid accuracy")
        plt.legend()
        plt.title("graph for "+data_type+" seprable classes on two hidden layers")
        plt.show()


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
    
    X_train, X_test, y_train_int, y_test_int = train_test_split(x, y_int, test_size=0.2)
    X_train, X_valid, y_train_int, y_valid_int = train_test_split(X_train, y_train_int, test_size=0.25)
    y_train = getY(y_train_int)
    y_test = getY(y_test_int)
    y_valid = getY(y_valid_int)
    p = twoHiddenLayers(3, 3, 1, 5, 7)
    # print(getattr(p,"w"))
    p.train(x, y, y_int, 1.0, 0, X_test, y_test_int, X_valid, y_valid_int, 10, "testing twoHiddenLayers")
        
    
    
    
    
    
    
    
    
    
    
    
    