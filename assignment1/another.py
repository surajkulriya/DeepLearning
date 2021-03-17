# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Mar 16 23:36:04 2021

# @author: surajkulriya
# """


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Mar 16 21:27:18 2021

# @author: surajkulriya
# """

import numpy as np
import matplotlib.pyplot as plt

# class perceptron:
w=[
    [0],
    [0],
    [0]
]
dimension=0
error = []
# def __init__(self,in_dimension):
#     self.dimension = in_dimension
#     self.w=[]
#     for i in range(in_dimension):
#         self.w.append([0])
def mean_sq_error(y, y_exp):
    ans = 0
    for i in range(len(y)):
        ans= ans+(y[i]-y_exp[i])**2
    ans=1.0*ans/len(y)
    # print("ans = ",ans)
    ans=(ans)**(0.5)
    return ans;

def output(x,w):
    return np.matmul(x,w)
        
        
def show_weigths(w):
    return w;

def train(w, x, y, learning_rate, momentum, error):
    max_epoch = 1000
    # print(momentum)
    weight=[]
    for i in range(len(x[0])): weight.append([0])
    error = []
    last_error = 1000
    epoch = 0
    last_delta = []
    for i in range(len(w)):
        last_delta.append(0);
    i=0;
    epoch=0
    for epoch in range(max_epoch):
        # print(epoch)
        y_arr=[]
        for i in range(len(x)):
            y_exp = np.matmul(x[i],weight)
            # if(i==20): print(y_exp)
            y_arr.append(y_exp)
            for j in range(len(weight)):
                # print(len(self.w))
                weight[j]+= (1.0)*(y[i]-y_exp)*(x[i][j]) 
                # print(delta)
                # last_delta[j] = delta
                # weight[j]+=delta
        error.append(mean_sq_error(y, y_arr))
        last_error = error[-1]
        epoch+=1
    w=weight
    return w, error
def errVsepoch():
    plt.plot(error)
    plt.show()
    
    
# if( __name__=="__main__"):
        

xw=[]
x=[]
y=[]
for i in range (100):
    x.append([i,2*i+1000,1])
    y.append([i*100])




# def train(w, x, y, learning_rate, momentum, error):
max_epoch = 1000
# print(momentum)
weight=[]
for i in range(len(x[0])): weight.append([float(0.0)])
weight = [
    [0],
    [0],
    [0]
    ]
error = []
last_error = 1000
epoch = 0
last_delta = []
for i in range(len(w)):
    last_delta.append(0);
i=0;
epoch=0
for epoch in range(max_epoch):
    # print(epoch)
    y_arr=[]
    for i in range(len(x)):
        y_exp = np.matmul(x[i],weight)
        # if(i==20): print(y_exp)
        y_arr.append(y_exp)
        for j in range(len(weight)):
            # print(len(self.w))
            weight[j]= weight[j]+(1.0)*(y[i]-y_exp)*(x[i][j])
            # print(delta)
            # last_delta[j] = delta
            # weight[j]+=delta
    error.append(mean_sq_error(y, y_arr))
    last_error = error[-1]
    # epoch+=1
w=weight

# p = perceptron(3);
# w,error = train(w,x,y,1.0,0, error)
errVsepoch()
print(output([555,2*555+1000,1],w))
# error=getattr(p,"error")
# print(error[0])
# print(error[-1])
print(show_weigths(w))
# p.errVsepoch()

# print(p.output([[5],[4],[3]]));


# def set_list(list): 
#     list = ["A", "B", "C"] 
#     return list
  
# def add(list): 
#     list.append("D") 
#     return list
  
# my_list = ["E"] 
  
# print(set_list(my_list)) 
  
# print(add(my_list)) 

# print(my_list)