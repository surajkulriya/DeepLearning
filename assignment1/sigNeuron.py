#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 23:22:27 2021

@author: surajkulriya
"""



import numpy as np
import matplotlib.pyplot as plt
import functionFiles as functions     #self defined library

func = functions.sigmoidActfunc
diffFunc = functions.diffSigmoidActfunc

# class sigPerceptron:

class sigPerceptron:
    w=[]
    dimension=0;
    error=[]
    beta = 1.0
    # actFunc = functions.sigmoidActfunc
    # diffActFunc = functions.diffSigmoidActfunc
    def __init__(self,in_dimension, Beta):
        self.beta = Beta
        self.dimension = in_dimension
        self.w=[]
        for i in range(in_dimension):
            self.w.append([float(1)])
    def mean_sq_error(self,y, y_exp):
        ans = 0
        for i in range(len(y)):
            ans+= (y[i]-y_exp[i])**2
        ans=1.0*ans/len(y)
        # print("ans = ",ans)
        ans=(ans)**(0.5)
        return ans;
    
    def actFunc(self,x):
        return functions.sigmoidActfunc(self.beta, x)
    
    def diffActFunc(self,x):
        return functions.diffSigmoidActfunc(self.beta, x)
    
    def output(self,x):             # here x is assumed to be a tuple
        a = np.matmul(x,self.w)
        # s = self.actFunc(self.beta,a)
        return a;
    
    def expClass(self,x):
        y = self.output(x)
        return self.actFunc(y)
        
    def show_weigths(self):
        return self.w;
    
    def train(self, x, y, learning_rate, momentum):
        max_epoch = 10
        epoch = 0
        last_delta = []
        for i in range(len(self.w)):
            last_delta.append(0);
        i=0;
        epoch=0
        for epoch in range(max_epoch):
            y_arr=[]
            for i in range(len(x)):
                y_exp = self.output(x[i])
                # print(y_exp)
                s_exp = self.actFunc(y_exp)
                y_arr.append(s_exp)
                for j in range(len(self.w)):
                    # uncomment below line for integer weights and don't forget to comment the line below its after that
                    # self.w[j]+=((y[i]-y_exp)*(x[i][j])) 
                    delta = learning_rate* ((y[i]-s_exp)*(x[i][j]))*(self.diffActFunc(y_exp)) 
                    self.w[j]+=delta + momentum*(last_delta[j])
                    last_delta[j] = delta
                    
                    
                    # np.add(self.w[j], (learning_rate)*((y[i]-y_exp)*(x[i][j])), out=self.w[j], casting="unsafe")
            self.error.append(self.mean_sq_error(y, y_arr))

    def errVsepoch(self):
        plt.plot(self.error)
        plt.show()
        
        
if( __name__=="__main__"):
    # this is just a test
    xw=[]
    x=[]
    y=[]
    for i in range (100):
        x.append([i,i,1])
        if(i<50):
            y.append([0])
        else:
            y.append([1])
    
    p = sigPerceptron(3,1);
    p.train(x,y,0.001,1)
    p.errVsepoch()
    print(p.expClass([25,25,1]))
    print(p.show_weigths())
    
    # print(func(1,-1))
    
    
    
    
    
    
    
    
    
