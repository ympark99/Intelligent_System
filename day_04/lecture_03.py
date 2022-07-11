#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:14:12 2022

@author: youngmin
"""

import numpy as np

X = np.arange(1, 11).reshape(-1,1)
y = [5,8,10,9,7,5,3,6,9,10]

import matplotlib.pyplot as plt

plt.plot(X, y, 'xb')
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,y)

print(model.score(X,y))





