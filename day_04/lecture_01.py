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

model = LinearRegression(n_jobs=-1)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'Train : {score}')

score = model.score(X_test, y_test)
print(f'Test : {score}')

X_test.iloc[:3, :2]
pred = model.predict(X_test.iloc[:1])
print(pred)

print(model.coef_)

print(model.intercept_)

# pred = 3.25 * model.coef_[0] + 39.0 * model.coef_[1] + 4.503205 * model.coef_[2] + \
#    1.073718 * model.coef_[3] + 1109.0 * model_coef_[4]

print(pred)


from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_absolute_percentage_error

from sklearn.metrics import mean_squared_error

pred = model.predict(X_train)


mae = mean_absolute_error(y_train, pred)
print(mae)

print(y_train.describe())

mape = mean_absolute_percentage_error(y_train, pred)
print(mape)


# 시험!!! 분석결과보여주고 
# 특정 컬럼에 대한 가중치가 다른 컬럼에 비해 상대적으로 높음
# 가중치의 절대값이 클 수록 영향력이 높은 특성!
# 중요도가 높다고 생각되는 거 선택 이유 서술형으로 작성
print(X.info())
print(model.coef_)

#ridge










