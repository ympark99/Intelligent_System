#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:09:25 2022

@author: youngmin
"""

# 회귀 분석
import pandas as pd
# load : 연습을 위한 간단한 데이터 셋
# fetch : 실 데이터 셋
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)

print(X.info())

print(X.isnull())
print(X.isnull().sum())

pd.options.display.max_columns = 100
print(X.describe(include = 'all'))

print(y.head())
print(y.tail())


# print(y.value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                         #           stratify=y,
                                                    random_state=1
                                                    )

print(X_train.shape, X_test.shape)
print(len(y_train), len(y_test))

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge, Lasso

lr_model = LinearRegression(n_jobs=-1)
ridge_model = Ridge(alpha=1.0, random_state=1)
lasso_model = Lasso(alpha=1.0, random_state=1)

score = lr_model.fit(X_train, y_train)
print(score)
score = ridge_model.fit(X_train, y_train)
print(score)
score = lasso_model.fit(X_train, y_train)
print(score)









