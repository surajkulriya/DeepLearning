"""

@author surajKulriya
"""

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Mar 16 23:36:04 2021

# @author: surajkulriya
# """


import numpy as np
import matplotlib.pyplot as plt

# class perceptron:

class perceptron:
    w=[]
    dimension=0;
    error=[]
    def __init__(self,in_dimension):
        self.dimension = in_dimension
        self.w=[]
        for i in range(in_dimension):
            self.w.append([float(0)])
    def mean_sq_error(self,y, y_exp):
        ans = 0
        for i in range(len(y)):
            ans+= (y[i]-y_exp[i])**2
        ans=1.0*ans/len(y)
        # print("ans = ",ans)
        ans=(ans)**(0.5)
        return ans;
    
    def output(self,x):
        return np.matmul(x,self.w)
            
            
    def show_weigths(self):
        return self.w;
    
    def train(self, x, y, learning_rate, momentum):
        max_epoch = 1000
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
                y_arr.append(y_exp)
                for j in range(len(self.w)):
                    # uncomment below line for integer weights and don't forget to comment the line below its after that
                    # self.w[j]+=((y[i]-y_exp)*(x[i][j])) 
                    self.w[j]+=learning_rate* ((y[i]-y_exp)*(x[i][j])) 
                    # np.add(self.w[j], (learning_rate)*((y[i]-y_exp)*(x[i][j])), out=self.w[j], casting="unsafe")
            self.error.append(self.mean_sq_error(y, y_arr))

    def errVsepoch(self):
        plt.plot(self.error)
        plt.show()
        
        
if( __name__=="__main__"):
    xw=[]
    x=[]
    y=[]
    for i in range (100):
        x.append([i,2*i+1000,1])
        y.append([i*100])
    
    p = perceptron(3);
    p.train(x,y,0.000001,0)
    p.errVsepoch()
    print(p.output([555,2*555+1000,1]))
    print(p.show_weigths())
    
    
    
    
    
    
    
    
    
    
    
    
    
