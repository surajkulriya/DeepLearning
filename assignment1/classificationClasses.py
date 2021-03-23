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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
class noHiddenLayer:
    inNeurons = 1;
    outNeurons = 1
    w = []
    beta = 1
    accuracy = []
    error = []
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
    def getOutput(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(len(self.w)):
                ans[i].append(np.matmul(x[i],self.w[j]))
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
    
        
    def fOut(self, x):
        ans = []
        for i in range(len(x)):
            ans.append(self.actFunc(x[i]))
        return ans;
    def getfOut(self, x):
        ans = []
        for i in range(len(x)):
            ans.append(self.fOut(x[i]))
        return ans;
    def mse(self, y_true, y_pred):
        ans = 0
        for i in range(len(y_pred)):
            for j in range(len(y_pred[i])):
                ans += (y_true[i][j]-y_pred[i][j])*(y_true[i][j]-y_pred[i][j])
        return ans/(2.0*len(y_true))
    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, max_epoch, data_type):
        test_mse = []
        valid_mse = []
        for epoch in range(max_epoch):
            y_arr = []
            y_s = []
            for i in range(len(x)):
                y_exp = self.output(x[i])
                
                y_arr.append(y_exp.index(max(y_exp))+1)
                s = []
                for ii in range(len(y_exp)): s.append(self.actFunc(y_exp[ii]))
                y_s.append(s)
                for k in range(len(s)):    # k'th output neuron
                    mul = (learning_rate)*(y[i][k]-s[k])*(self.diffActFunc(y_exp[k]))
                    for j in range(len(self.w[k])):  #weight associated with j'th neuron of 1st layer and k'th neuron of 2nd layer
                        self.w[k][j]+= mul*x[i][j]
        
            (self.accuracy).append(self.classAcuracy(y_int, y_arr))
            self.error.append(self.mse(y_s, y))
            y_test_exp = self.getOutput(x_test)
            y_valid_exp = self.getOutput(x_valid)
            y_test_s = self.getfOut(y_test_exp)
            y_valid_s = self.getfOut(y_valid_exp)
            test_mse.append(self.mse(y_test, y_test_s))
            valid_mse.append(self.mse(y_valid, y_valid_s))
        # print(y_arr)
        plt.plot(self.error, label = "train error")
        plt.plot(test_mse, label = "test error")
        plt.plot(valid_mse, label = "valid error")
        plt.title("graph for "+data_type+" seprable classes on no hidden layers")
        plt.legend()
        plt.show()
    
    def classAcuracyVsepoch(self):
        plt.plot(self.accuracy)

    def modelVStarget(self, x, y, data_type, data_sep):
        y_pred = self.getClassLabels(x)
        
        x_x = []
        x_y = []
        for i in range(len(x)):
            x_x.append(x[i][0])
            x_y.append(x[i][1])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("scatter plot for Y true for " + data_sep+" " + data_type+"for no hidden layer");
        ax.scatter3D(x_x, x_y, y, c = y, label = "Y true")
        plt.legend()
        plt.show()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("scatter plot for Y  pred for " + data_sep+" " + data_type+"for no hidden layer");
        ax.scatter3D(x_x, x_y, y_pred, c = y_pred, label = "y pred")
        plt.legend()
        plt.show()
    def confMat(self, x, y):
        y_pred = self.getClassLabels(x)
        cf_matrix = confusion_matrix(y, y_pred)
        sns.heatmap(cf_matrix, annot=True, fmt="d")
        plt.show()


class oneHiddenLayer:
    inNeuron = 1
    outNeurons = 1
    nHL1 = 1
    beta = 1
    w = [[],[]]
    accuracy = []
    error = []
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
    
    def getMSE(self, y_true, y_pred):
        ans = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                ans += (y_true[i][j]-y_pred[i][j])*(y_true[i][j]-y_pred[i][j])/2.0
        return ans/len(y_pred)
    
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
    
    def getOutputDL1(self, x):
        ans = []
        for tuplex in range(len(x)):
            ans.append([])
            for neuron in range(self.nHL1):
                ans[tuplex].append(np.matmul(x[tuplex], self.w[0][neuron]))
        return ans;
    
    def getSoutDL1(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(len(x[i])):
                ans[i].append(self.actFunc(x[i][j]))
        return ans
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
    
    def getFactOut(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(self.outNeurons):
                ans[i].append(np.matmul(x[i], self.w[1][j]))
        return ans
    def fout(self, x): # final output of output layer
        ans = []
        for i in range(len(x)):
            ans.append(self.outActFunc(x[i]))
        return ans 
    
    def getFout(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(len(x[i])):
                ans[i].append(self.actFunc(x[i][j]))
        return ans;
    
    def getNetOut(self, x):
        out1 = self.getOutputDL1(x)
        sout1 = self.getSoutDL1(out1)
        for i in range(len(sout1)): sout1[i].append(float(1))
        outo = self.getFactOut(sout1)
        outs = self.getFout(outo)
        return outs
    
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
    def plotHiddenLayer(self, x_train, y_train, epoch, data_type, data_sep):
        if(len(x_train[0])==3):
            for neuron in range(self.nHL1):
                output = []
                x_train_x = []
                x_train_y = []
                for tuplex in range(len(x_train)):
                    output.append(np.matmul(x_train[tuplex], self.w[0][neuron]))
                    x_train_x.append(x_train[tuplex][0])
                    x_train_y.append(x_train[tuplex][1])
                    output[tuplex] = self.actFunc(output[tuplex])
                ax = plt.axes(projection='3d')
                ax.scatter3D(x_train_x, x_train_y, output, c = output)
                ax.set_title('surface plot for hidden layer for '+data_sep+' '+ data_type+ ' for '+str(epoch)+"'th epoch for "+str(neuron)+"'th neuron");
                plt.show()
                
    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, max_epoch, data_type):
        test_mse = []
        valid_mse = []
        for epoch in range(max_epoch):
            y_arr = []
            y_s = []
            if(epoch%(max_epoch//3)==0):
                    self.plotHiddenLayer(x, y, epoch, "train data","non-linearly seprable")
                    self.plotHiddenLayer(x_test, y_test, epoch, "test data","non-linearly seprable")
                    self.plotHiddenLayer(x_valid, y_valid, epoch, "validation data","non-linearly seprable")
            for tuplex in range(len(x)):
                a1 = self.outputDL1(x[tuplex])
                s1 = self.sOut(a1)
                s1.append(float(1.0)) #bias term
                a2 = self.factOut(s1)
                s2 = self.fout(a2)
                y_s.append(s2)
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
            self.error.append(self.getMSE(y, y_s))
            y_test_exp = self.getNetOut(x_test)
            y_valid_exp = self.getNetOut(x_valid)
            test_mse.append(self.getMSE(y_test, y_test_exp))
            valid_mse.append(self.getMSE(y_valid, y_valid_exp))
        # print(y_arr)
        plt.plot(self.error, label = "train MSE")
        plt.plot(test_mse, label = "test MSE")
        plt.plot(valid_mse, label = "valid MSE")
        plt.legend()
        plt.title("graph for "+data_type+" seprable classes on one hidden layer")
        plt.show()
    def modelVStarget(self, x, y, data_type, data_sep):
        y_pred = self.getClassLabels(x)
        ax = plt.axes(projection='3d')
        x_x = []
        x_y = []
        for i in range(len(x)):
            x_x.append(x[i][0])
            x_y.append(x[i][1])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("scatter plot for Y true for " + data_sep+" " + data_type+"for one hidden layer");
        ax.scatter3D(x_x, x_y, y, c = y, label = "Y true")
        plt.legend()
        plt.show()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("scatter plot for Y  pred for " + data_sep+" " + data_type+"for one hidden layer");
        ax.scatter3D(x_x, x_y, y_pred, c = y_pred, label = "y pred")
        plt.legend()
        plt.show()
    def confMat(self, x, y):
        y_pred = self.getClassLabels(x)
        cf_matrix = confusion_matrix(y, y_pred)
        sns.heatmap(cf_matrix, annot=True, fmt="d")
        plt.show()


    
class twoHiddenLayers:
    inNeuron = 1
    outNeurons = 1
    nHL1 = 1
    nHL2 = 1
    beta = 1
    w = [[],[], []]
    accuracy = []
    error = []
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
    def getMSE(self, y_true, y_pred):
        ans = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                ans += (y_true[i][j]-y_pred[i][j])*(y_true[i][j]-y_pred[i][j])/2.0
        return ans/len(y_pred)
    
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
    
    def getOutputDL1(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(self.nHL1):
                ans[i].append(np.matmul(x[i], self.w[0][j]))
        return ans;
    
    def getSoutDL1(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(len(x[i])):
                ans[i].append(self.actFunc(x[i][j]))
        return ans;
    
    def outputDL2(self, x):  # activation values output of of hidden layer 2
        outDLlayer2 = []   #output of Hidden Layer
        for i in range(self.nHL2):
            outDLlayer2.append(np.matmul(x, self.w[1][i]))
        return outDLlayer2
    
    def getOutputDL2(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(self.nHL2):
                ans[i].append(np.matmul(x[i], self.w[1][j]))
        return ans;
    
    def getSoutDL2(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(len(x[i])):
                ans[i].append(self.actFunc(x[i][j]))
        return ans;
        
    
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
    def getFactout(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(self.outNeurons):
                ans[i].append(np.matmul(x[i], self.w[2][j]))
        return ans;
    def getSoutLayer(self, x):
        ans = []
        for i in range(len(x)):
            ans.append([])
            for j in range(len(x[i])):
                ans[i].append(self.actFunc(x[i][j]))
        return ans
    def getNetOut(self, x):
        a1 = self.getOutputDL1(x)
        s1 = self.getSoutDL1(a1)
        for i in range(len(s1)): s1[i].append(float(1))
        a2 = self.getOutputDL2(s1)
        s2 = self.getSoutDL2(a2)
        for i in range(len(s2)): s2[i].append(float(1))
        ao = self.getFactout(s2)
        ans = self.getSoutLayer(ao)
        return ans;
    
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
    
    def plotHiddenLayer(self, x_train, y_train, epoch, data_type, data_sep):
        if(len(x_train[0])==2):
            output1 = []
            for neuron in range(self.nHL1):
                output1.append([])
                output = []
                x_train_x = []
                for tuplex in range(len(x_train)):
                    output.append(np.matmul(x_train[tuplex], self.w[0][neuron]))
                    x_train_x.append(x_train[tuplex][0])
                    output[tuplex] = self.actFunc(output[tuplex])
                    output1[neuron].append(output[tuplex])
                plt.scatter(x_train_x, output)
                plt.title("output of hidden layer 1 for univariate data for "+str(epoch)+"'th epoch for "+data_type+"for "+str(neuron)+"'th neuron")
                plt.xlabel("input X values")
                plt.ylabel("output of Hidden neuron")
                plt.show()
            output1.append([])
            for i in range(len(x_train)): output1[-1].append(1)
            output1 = np.transpose(output1)
            for neuron in range(self.nHL2):
                output = []
                x_train_x = []
                x_train_y = []
                for tuplex in range(len(output1)):
                    output.append(np.matmul(output1[tuplex], self.w[1][neuron]))
                    x_train_x.append(x_train[tuplex][0])
                    output[tuplex] = self.actFunc(output[tuplex])
                plt.scatter(x_train_x, output)
                plt.title("output of hidden layer 2 for"+ data_sep+ " for "+str(epoch)+"'th epoch for "+data_type+"for "+str(neuron)+"'th neuron")
                plt.xlabel("input X values")
                plt.ylabel("output of Hidden neuron")
                plt.show()
        if(len(x_train[0])==3):
            output1 = []
            for neuron in range(self.nHL1):
                output = []
                output1.append([])
                x_train_x = []
                x_train_y = []
                for tuplex in range(len(x_train)):
                    output.append(np.matmul(x_train[tuplex], self.w[0][neuron]))
                    x_train_x.append(x_train[tuplex][0])
                    x_train_y.append(x_train[tuplex][1])
                    output[tuplex] = self.actFunc(output[tuplex])
                    output1[neuron].append(output[tuplex])
                ax = plt.axes(projection='3d')
                ax.scatter3D(x_train_x, x_train_y, output, c = output)
                ax.set_title('surface plot for hidden layer 1 for bivariate data for '+ data_type+ ' for '+str(epoch)+"'th epoch for "+str(neuron)+"'th neuron");
                plt.show()
            output1.append([])
            for i in range(len(x_train)):output1[-1].append(1)
            output1 = np.transpose(output1)
            for neuron in range(self.nHL2):
                output = []
                x_train_x = []
                x_train_y = []
                for tuplex in range(len(output1)):
                    output.append(np.matmul(output1[tuplex], self.w[1][neuron]))
                    x_train_x.append(x_train[tuplex][0])
                    x_train_y.append(x_train[tuplex][1])
                    output[tuplex] = self.actFunc(output[tuplex])
                ax = plt.axes(projection='3d')
                ax.scatter3D(x_train_x, x_train_y, output, c = output)
                ax.set_title('surface plot for hidden layer 2 for bivariate data for '+ data_type+ ' for '+str(epoch)+"'th epoch for "+str(neuron)+"'th neuron");
                plt.show()
    
    
    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, max_epoch, data_type):
        test_mse = []
        valid_mse = []
        for epoch in range(max_epoch):
            y_arr = []
            y_s = []
            if(epoch%(max_epoch//3)==0):
                    self.plotHiddenLayer(x, y, epoch, "train data",data_type)
                    self.plotHiddenLayer(x_test, y_test, epoch, "test data",data_type)
                    self.plotHiddenLayer(x_valid, y_valid, epoch, "validation data",data_type)
            
            for tuplex in range(len(x)):
                a1 = self.outputDL1(x[tuplex])
                s1 = self.sOut(a1)
                s1.append(float(1.0)) #bias term
                a2 = self.outputDL2(s1)
                s2 = self.sOut(a2)
                s2.append(float(1.0)) #bias term
                a3 = self.factOut(s2)
                s3 = self.fout(a3)
                y_s.append(s3)
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
            y_test_exp = self.getNetOut(x_test)
            # print(y_test_exp)
            y_valid_exp = self.getNetOut(x_valid)
            test_mse.append(self.getMSE(y_test, y_test_exp))
            valid_mse.append(self.getMSE(y_valid, y_valid_exp))
        # print(y_arr)
        plt.plot(self.error, label = "train accuracy")
        plt.plot(test_mse, label = "test accuracy")
        plt.plot(valid_mse, label = "valid accuracy")
        plt.legend()
        plt.title("graph for "+data_type+" seprable classes on two hidden layers")
        plt.show()
    def modelVStarget(self, x, y, data_type, data_sep):
        y_pred = self.getClassLabels(x)
        ax = plt.axes(projection='3d')
        x_x = []
        x_y = []
        for i in range(len(x)):
            x_x.append(x[i][0])
            x_y.append(x[i][1])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("scatter plot for Y true for " + data_sep+" " + data_type+"for two hidden layer");
        ax.scatter3D(x_x, x_y, y, c = y, label = "Y true")
        plt.legend()
        plt.show()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("scatter plot for Y  pred for " + data_sep+" " + data_type+"for two hidden layer");
        ax.scatter3D(x_x, x_y, y_pred, c = y_pred, label = "y pred")
        plt.legend()
        plt.show()
    def confMat(self, x, y):
        y_pred = self.getClassLabels(x)
        cf_matrix = confusion_matrix(y, y_pred)
        sns.heatmap(cf_matrix, annot=True, fmt="d")
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
        
    
    
    
    
    
    
    
    
    
    
    
    