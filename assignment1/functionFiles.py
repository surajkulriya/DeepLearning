#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 23:27:04 2021

@author: surajkulriya
"""

import numpy as np

def sigmoidActfunc(b,x):
    return 1/(1 + np.exp(-b*float(x))) 

def tanh(b,x):
    return np.tanh(b*float(x))

def diffSigmoidActfunc(b,x):
    y = sigmoidActfunc(b,x)
    return b*(y*(1-y))

def diffTanh(b,x):
    y = tanh(b,x)
    return b*(1-y*y)

def LinearFunc(b,x):
    return b*float(x)

def diffLinearFunc(b,x):
    return b;