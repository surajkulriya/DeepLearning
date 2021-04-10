#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 00:27:05 2021

@author: surajkulriya
"""


import numpy as np
import functionFiles  #self defined library
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    def xVSy(self, x_train, y_train, x_test, y_test, x_valid, y_valid, graph_title_add ):
        if(len(x_train[0])==2):
            y_train_exp = self.getValues(x_train)
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            x_train_x = []
            x_test_x = []
            x_valid_x = []
            for i in range(len(x_train)): x_train_x.append(x_train[i][0])
            for i in range(len(x_test)): x_test_x.append(x_test[i][0])
            for i in range(len(x_valid)): x_valid_x.append(x_valid[i][0])
            
            plt.scatter(x_train_x,y_train, label = "true Y train")
            plt.scatter(x_train_x,y_train_exp, label = "model Y train")
            plt.title("train data for no hidden layers for univariate data"+graph_title_add)
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
            
            plt.scatter(x_test_x,y_test, label = "true Y test")
            plt.scatter(x_test_x,y_test_exp, label = "model Y test")
            plt.title("test data for no hidden layers for univariate data"+graph_title_add)
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
            
            plt.scatter(x_valid_x,y_valid, label = "true Y valid")
            plt.scatter(x_valid_x,y_valid_exp, label = "model Y valid")
            plt.title("validation data for no hidden layers for univariate data"+graph_title_add)
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
        elif(len(x_train[0])==3):
            y_train_exp = self.getValues(x_train)
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            x_train_x = []; x_train_y = []
            x_test_x = []; x_test_y = []
            x_valid_x = []; x_valid_y = []
            
            for i in range(len(x_train)): x_train_x.append(x_train[i][0])
            for i in range(len(x_test)): x_test_x.append(x_test[i][0])
            for i in range(len(x_valid)): x_valid_x.append(x_valid[i][0])
            
            for i in range(len(x_train)): x_train_y.append(x_train[i][1])
            for i in range(len(x_test)): x_test_y.append(x_test[i][1])
            for i in range(len(x_valid)): x_valid_y.append(x_valid[i][1])
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_train_x, x_train_y, np.transpose(y_train)[0], label = "Y_train true")
            ax.scatter3D(x_train_x, x_train_y, np.transpose(y_train_exp)[0], label = "Y_train model")
            plt.title("graph for train data for no hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_test_x, x_test_y, np.transpose(y_test)[0], label = "Y_test true")
            ax.scatter3D(x_test_x, x_test_y, np.transpose(y_test_exp)[0], label = "Y_test model")
            plt.title("graph for test data for no hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_valid_x, x_valid_y, np.transpose(y_valid)[0], label = "Y_validation true")
            ax.scatter3D(x_valid_x, x_valid_y, np.transpose(y_valid_exp)[0], label = "Y_validation model")
            plt.title("graph for validation data for no hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
    
    def plotOutLayer(self, x, y, epoch, data_type):
        if(len(x[0])==2):
            x_x = []
            for i in range(len(x)): x_x.append(x[i][0])
            for neuron in range(self.outNeurons):
                output = []
                for tuplex in range(len(x)):
                    output.append(np.matmul(x[tuplex], self.w[neuron]))
                    output[tuplex] = self.actFunc(output[tuplex])
                plt.scatter(x_x, output)
                plt.title("output of output layer for univariate data for "+str(epoch)+"'th epoch for "+data_type+"for "+str(neuron)+"'th neuron")
                plt.xlabel("input X values")
                plt.ylabel("output of output neuron")
                plt.show()
        if(len(x[0])==3):
            x_x = []
            x_y = []
            for i in range(len(x)):
                x_x.append(x[i][0])
                x_y.append(x[i][1])
            for neuron in range(self.outNeurons):
                output = []
                for tuplex in range(len(x)):
                    output.append(np.matmul(x[tuplex], self.w[neuron]))
                    output[tuplex] = self.actFunc(output[tuplex])
                ax = plt.axes(projection='3d')
                ax.scatter3D(x_x, x_y, output, c = output)
                ax.set_title('surface plot for output layer for bivariate data for '+ data_type+ ' for '+str(epoch)+"'th epoch for "+str(neuron)+"'th neuron");
                plt.show()
                
    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, min_error, max_epoch, data_type):
        test_rms = []
        valid_rms= []
        epoch = 0
        error = 10000
        while(epoch<max_epoch and error>min_error):
            y_arr = []
            if(epoch%(max_epoch//3)==0):
                    self.plotOutLayer(x, y, epoch, "train ")
                    self.plotOutLayer(x_test, y_test, epoch, "test ")
                    self.plotOutLayer(x_valid, y_valid, epoch, "valid ")
            
            for i in range(len(x)):
                y_exp = self.output(x[i])
                y_arr.append(y_exp)
                s = []
                for ii in range(len(y_exp)): s.append(self.actFunc(y_exp[ii]))
                    
                for k in range(len(s)):    # k'th output neuron
                    mul = (learning_rate)*(y[i][k]-s[k])*(self.diffActFunc(y_exp[k]))
                    for j in range(len(self.w[k])):  #weight associated with j'th neuron of 1st layer and k'th neuron of 2nd layer
                        self.w[k][j]+= mul*x[i][j]
        
            (self.rms_arr).append(self.rms(y, y_arr))
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            test_rms.append(self.rms(y_test, y_test_exp))
            valid_rms.append(self.rms(y_valid, y_valid_exp))
            epoch+=1
            error = self.rms_arr[-1]
        # print(y_arr)
        plt.plot(self.rms_arr, label = "train rms")
        plt.plot(test_rms, label = "test rms")
        plt.plot(valid_rms, label = "valid rms")
        plt.title("graph for "+data_type+" regression on no hidden layers")
        plt.legend()
        plt.show()
    
    def modelVSexact(self, x, y, data_type):
        y_exp = self.getValues(x)
        plt.scatter(y, y_exp)
        plt.title("scatter plot for "+ data_type+" on no hidden layer")
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
    def xVSy(self, x_train, y_train, x_test, y_test, x_valid, y_valid, graph_title_add ):
        if(len(x_train[0])==2):
            y_train_exp = self.getValues(x_train)
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            x_train_x = []
            x_test_x = []
            x_valid_x = []
            for i in range(len(x_train)): x_train_x.append(x_train[i][0])
            for i in range(len(x_test)): x_test_x.append(x_test[i][0])
            for i in range(len(x_valid)): x_valid_x.append(x_valid[i][0])
            
            plt.scatter(x_train_x,y_train, label = "true Y train")
            plt.scatter(x_train_x,y_train_exp, label = "model Y train")
            plt.title("train data for one hidden layers for univariate data"+graph_title_add)
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
            
            plt.scatter(x_test_x,y_test, label = "true Y test")
            plt.scatter(x_test_x,y_test_exp, label = "model Y test")
            plt.title("test data for one hidden layers for univariate data"+graph_title_add)
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
            
            plt.scatter(x_valid_x,y_valid, label = "true Y valid")
            plt.scatter(x_valid_x,y_valid_exp, label = "model Y valid")
            plt.title("validation data for one hidden layers for univariate data"+graph_title_add)
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
        elif(len(x_train[0])==3):
            y_train_exp = self.getValues(x_train)
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            x_train_x = []; x_train_y = []
            x_test_x = []; x_test_y = []
            x_valid_x = []; x_valid_y = []
            
            for i in range(len(x_train)): x_train_x.append(x_train[i][0])
            for i in range(len(x_test)): x_test_x.append(x_test[i][0])
            for i in range(len(x_valid)): x_valid_x.append(x_valid[i][0])
            
            for i in range(len(x_train)): x_train_y.append(x_train[i][1])
            for i in range(len(x_test)): x_test_y.append(x_test[i][1])
            for i in range(len(x_valid)): x_valid_y.append(x_valid[i][1])
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_train_x, x_train_y, np.transpose(y_train)[0], label = "Y_train true")
            ax.scatter3D(x_train_x, x_train_y, np.transpose(y_train_exp)[0], label = "Y_train model")
            plt.title("graph for train data for one hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_test_x, x_test_y, np.transpose(y_test)[0], label = "Y_test true")
            ax.scatter3D(x_test_x, x_test_y, np.transpose(y_test_exp)[0], label = "Y_test model")
            plt.title("graph for test data for one hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_valid_x, x_valid_y, np.transpose(y_valid)[0], label = "Y_validation true")
            ax.scatter3D(x_valid_x, x_valid_y, np.transpose(y_valid_exp)[0], label = "Y_validation model")
            plt.title("graph for validation data for one hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
    
    def plotHiddenLayer(self, x_train, y_train, epoch, data_type):
        # if(len(x_train[0])==2):
        #     x_train_x = []
        #     outDL1_train = []
        #     for i in range(len(x_train)): x_train_x.append(x_train[i][0])
        #     for i in range(len(x_train)):
        #         a1 = (self.outputDL1(x_train[i]))
        #         outDL1_train.append(self.sOut(a1))
            
        #     x_train_plot = []
        #     x_train_HL_plot = []
        #     for i in range(len(x_train_x)):
        #         for j in range(len(outDL1_train[i])):
        #             x_train_plot.append(x_train_x[i])
        #             x_train_HL_plot.append(outDL1_train[i][j])
            
        #     plt.scatter(x_train_plot, x_train_HL_plot)
        #     plt.title("output of hidden layer 1 for univariate data for "+str(epoch)+"'th epoch for "+data_type)
        #     plt.xlabel("input X values")
        #     plt.ylabel("output of Hidden Layer")
        #     plt.show()
        if(len(x_train[0])==2):
            output1 = []            
            for neuron in range(self.nHL1):
                output = []
                output1.append([])
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
            for neuron in range(self.outNeurons):
                output = []
                for tuplex in range(len(output1)):
                    output.append(np.matmul(output1[tuplex], self.w[1][neuron]))
                    output[tuplex] = self.outActFunc(output[tuplex])
                plt.scatter(x_train_x, output)
                plt.title("output of output layer for univariate data for "+str(epoch)+"'th epoch for "+data_type+"for "+str(neuron)+"'th neuron")
                plt.xlabel("input X values")
                plt.ylabel("output of output neuron")
                plt.show()
        if(len(x_train[0])==3):
            output1 = []
            for neuron in range(self.nHL1):
                output = []
                x_train_x = []
                output1.append([])
                x_train_y = []
                for tuplex in range(len(x_train)):
                    output.append(np.matmul(x_train[tuplex], self.w[0][neuron]))
                    x_train_x.append(x_train[tuplex][0])
                    x_train_y.append(x_train[tuplex][1])
                    output[tuplex] = self.actFunc(output[tuplex])
                    output1[neuron].append(output[tuplex])
                ax = plt.axes(projection='3d')
                ax.scatter3D(x_train_x, x_train_y, output, c = output)
                ax.set_title('surface plot for hidden layer for bivariate data for '+ data_type+ ' for '+str(epoch)+"'th epoch for "+str(neuron)+"'th neuron");
                plt.show()
            output1.append([])
            for i in range(len(x_train)): output1[-1].append(1)
            output1 = np.transpose(output1)
            for neuron in range(self.outNeurons):
                output = []
                for tuplex in range(len(output1)):
                    output.append(np.matmul(output1[tuplex], self.w[1][neuron]))
                    output[tuplex] = self.outActFunc(output[tuplex])
                ax = plt.axes(projection='3d')
                ax.scatter3D(x_train_x, x_train_y, output, c = output)
                ax.set_title('surface plot for output layer for bivariate data for '+ data_type+ ' for '+str(epoch)+"'th epoch for "+str(neuron)+"'th neuron");
                plt.show()
        # if(len(x_train[0])==3):
        #     outDL1_train = []
        #     for i in range(len(x_train)):
        #         a1 = (self.outputDL1(x_train[i]))
        #         outDL1_train.append(self.sOut(a1))
            
            
            
        #     x_trainx_plot = []
        #     x_trainy_plot = []
        #     x_train_HL_plot = []
        #     for i in range(len(x_train)):
        #         for j in range(len(outDL1_train[i])):
        #             x_trainx_plot.append(x_train[i][0])
        #             x_trainy_plot.append(x_train[i][1])
        #             x_train_HL_plot.append(outDL1_train[i][j])
            
        #     ax = plt.axes(projection='3d')
        #     ax.scatter3D(x_trainx_plot, x_trainy_plot, x_train_HL_plot, c = x_train_HL_plot)
        #     ax.set_title('surface plot for hidden layer for bivariate data for '+ data_type+ ' for '+str(epoch)+"'th epoch");
        #     plt.show()
            
    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid, min_error, max_epoch, data_type):
        test_rms= []
        valid_rms= []
        epoch_arr  = []
        
        epoch = 0
        error = 10000
        while(epoch < max_epoch and error>min_error):
            y_arr = []
            if(epoch%(max_epoch//3)==0):
                    self.plotHiddenLayer(x, y, epoch, "train data")
                    # self.plotHiddenLayer(x_test, y_test, epoch, "test data")
                    # self.plotHiddenLayer(x_valid, y_valid, epoch, "validation data")
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
            epoch_arr.append(epoch)
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            test_rms.append(self.rms(y_test, y_test_exp))
            valid_rms.append(self.rms(y_valid, y_valid_exp))
            epoch+=1
            error = self.rms_arr[-1]
        # print(y_arr)
        plt.plot(epoch_arr,self.rms_arr, label = "train rms")
        plt.plot(epoch_arr,test_rms, label = "test rms")
        plt.plot(epoch_arr,valid_rms, label = "valid rms")
        plt.legend()
        plt.title("graph for "+data_type+" regression on one hidden layer")
        plt.show()
        
    def modelVSexact(self, x, y, data_type):
        y_exp = self.getValues(x)
        plt.scatter(y, y_exp)
        plt.title("scatter plot for "+ data_type+" on one hidden layer")
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
    def xVSy(self, x_train, y_train, x_test, y_test, x_valid, y_valid, graph_title_add ):
        if(len(x_train[0])==2):
            y_train_exp = self.getValues(x_train)
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            x_train_x = []
            x_test_x = []
            x_valid_x = []
            for i in range(len(x_train)): x_train_x.append(x_train[i][0])
            for i in range(len(x_test)): x_test_x.append(x_test[i][0])
            for i in range(len(x_valid)): x_valid_x.append(x_valid[i][0])
            
            plt.scatter(x_train_x,y_train, label = "true Y train")
            plt.scatter(x_train_x,y_train_exp, label = "model Y train")
            plt.title("train data for two hidden layers for univariate data"+graph_title_add)
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
            
            plt.scatter(x_test_x,y_test, label = "true Y test")
            plt.scatter(x_test_x,y_test_exp, label = "model Y test")
            plt.title("test data for two hidden layers for univariate data"+graph_title_add)
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
            
            plt.scatter(x_valid_x,y_valid, label = "true Y valid")
            plt.scatter(x_valid_x,y_valid_exp, label = "model Y valid")
            plt.title("validation data for two hidden layers for univariate data" + graph_title_add )
            plt.xlabel("input data")
            plt.ylabel("output value")
            plt.legend()
            plt.show()
        elif(len(x_train[0])==3):
            y_train_exp = self.getValues(x_train)
            y_test_exp = self.getValues(x_test)
            y_valid_exp = self.getValues(x_valid)
            x_train_x = []; x_train_y = []
            x_test_x = []; x_test_y = []
            x_valid_x = []; x_valid_y = []
            
            for i in range(len(x_train)): x_train_x.append(x_train[i][0])
            for i in range(len(x_test)): x_test_x.append(x_test[i][0])
            for i in range(len(x_valid)): x_valid_x.append(x_valid[i][0])
            
            for i in range(len(x_train)): x_train_y.append(x_train[i][1])
            for i in range(len(x_test)): x_test_y.append(x_test[i][1])
            for i in range(len(x_valid)): x_valid_y.append(x_valid[i][1])
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_train_x, x_train_y, np.transpose(y_train)[0], label = "Y_train true")
            ax.scatter3D(x_train_x, x_train_y, np.transpose(y_train_exp)[0], label = "Y_train model")
            plt.title("graph for train data for two hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_test_x, x_test_y, np.transpose(y_test)[0], label = "Y_test true")
            ax.scatter3D(x_test_x, x_test_y, np.transpose(y_test_exp)[0], label = "Y_test model")
            plt.title("graph for test data for two hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(x_valid_x, x_valid_y, np.transpose(y_valid)[0], label = "Y_validation true")
            ax.scatter3D(x_valid_x, x_valid_y, np.transpose(y_valid_exp)[0], label = "Y_validation model")
            plt.title("graph for validation data for two hidden layers for bivariate data"+graph_title_add)
            plt.legend()
            plt.show()
    def plotHiddenLayer(self, x_train, y_train, epoch, data_type):
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
            output2 = []
            for neuron in range(self.nHL2):
                output = []
                x_train_x = []
                x_train_y = []
                output2.append([])
                for tuplex in range(len(output1)):
                    output.append(np.matmul(output1[tuplex], self.w[1][neuron]))
                    x_train_x.append(x_train[tuplex][0])
                    output[tuplex] = self.actFunc(output[tuplex])
                    output2[neuron].append(output[tuplex])
                plt.scatter(x_train_x, output)
                plt.title("output of hidden layer 2 for univariate data for "+str(epoch)+"'th epoch for "+data_type+"for "+str(neuron)+"'th neuron")
                plt.xlabel("input X values")
                plt.ylabel("output of Hidden neuron")
                plt.show()
            output2.append([])
            for i in range(len(output1)): output2[-1].append(1)
            output2 = np.transpose(output2)
            for neuron in range(self.outNeurons):
                output = []
                for tuplex in range(len(output2)):
                    output.append(np.matmul(output2[tuplex], self.w[2][neuron]))
                    output[tuplex] = self.outActFunc(output[tuplex])
                plt.scatter(x_train_x, output)
                plt.title("output of output layer for univariate data for "+str(epoch)+"'th epoch for "+data_type+"for "+str(neuron)+"'th neuron")
                plt.xlabel("input X values")
                plt.ylabel("output of output neuron")
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
            output2 = []
            for i in range(len(x_train)):output1[-1].append(1)
            output1 = np.transpose(output1)
            for neuron in range(self.nHL2):
                output = []
                x_train_x = []
                x_train_y = []
                output2.append([])
                for tuplex in range(len(output1)):
                    output.append(np.matmul(output1[tuplex], self.w[1][neuron]))
                    x_train_x.append(x_train[tuplex][0])
                    x_train_y.append(x_train[tuplex][1])
                    output[tuplex] = self.actFunc(output[tuplex])
                    output2[neuron].append(output[tuplex])
                ax = plt.axes(projection='3d')
                ax.scatter3D(x_train_x, x_train_y, output, c = output)
                ax.set_title('surface plot for hidden layer 2 for bivariate data for '+ data_type+ ' for '+str(epoch)+"'th epoch for "+str(neuron)+"'th neuron");
                plt.show()
            output2.append([])
            for i in range(len(output1)): output2[-1].append(1)
            output2 = np.transpose(output2)
            for neuron in range(self.outNeurons):
                output = []
                for tuplex in range(len(output2)):
                    output.append(np.matmul(output2[tuplex], self.w[2][neuron]))
                    output[tuplex] = self.outActFunc(output[tuplex])
                ax = plt.axes(projection='3d')
                ax.scatter3D(x_train_x, x_train_y, output, c = output)
                ax.set_title('surface plot for output layer for bivariate data for '+ data_type+ ' for '+str(epoch)+"'th epoch for "+str(neuron)+"'th neuron");
                plt.show()
    def train(self, x, y, y_int, learning_rate, momentum, x_test, y_test, x_valid, y_valid,min_error, max_epoch, data_type):
        test_rms= []
        valid_rms= []
        epoch = 0
        error = 10000
        while(error>min_error and epoch<max_epoch):
            y_arr = []
            if(epoch%(max_epoch//3)==0):
                    self.plotHiddenLayer(x, y, epoch, "train data")
                    # self.plotHiddenLayer(x_test, y_test, epoch, "test data")
                    # self.plotHiddenLayer(x_valid, y_valid, epoch, "validation data")
                    
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
            epoch+=1
            error = self.rms_arr[-1]
        # print(y_arr)
        plt.plot(self.rms_arr, label = "train rms")
        plt.plot(test_rms, label = "test rms")
        plt.plot(valid_rms, label = "valid rms")
        plt.legend()
        plt.title("graph for "+data_type+" regression on two hidden layers")
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
        
