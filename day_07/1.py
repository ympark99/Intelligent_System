#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:06:26 2022

@author: youngmin
"""

import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 10

fname_input = './titanic.csv'
data = pd.read_csv(fname_input,
                   header='infer', # 없으면 none
                    sep=',')

print(data.head())

print(data.info())

print(data.PassengerId.value_counts())

print(data.describe(include=object))
# 중복 없으면 주의
# cabin 결측 많음

data2 = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'],
                  inplace=False)


print(data2.info())
print(data2.isnull().sum())

data3 = data2.dropna(subset=['Age','Embarked'])
print(data3.info())
print(data3.isnull().sum())


print(data.Sex.value_counts())

print(data.Embarked.value_counts())


X = data3.iloc[:,1:]
y = data3.Survived

X_num = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
X_obj = [cname for cname in X.columns if X[cname].dtype not in ['int64', 'float64']]

print(X['Pclass'].dtype)
print(X['Age'].dtype)
print(X['Sex'].dtype)

X_num = X[X_num]
X_obj = X[X_obj]

print(X_num.info())
print(X_obj.info())

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse = False,
                        handle_unknown='ignore')


X_obj_encoded = encoder.fit_transform(X_obj)
print(X_obj.head())
print(X_obj_encoded[:5])

print(encoder.categories_)
print(encoder.feature_names_in_)

cols_encoded = [cname for cname in encoder.categories_]

X_obj_encoded = pd.DataFrame(X_obj_encoded,
                             columns=['s_f','s_m',
                                      'e_C','e_Q','e_S'])

print(X_obj_encoded)

print(X_num.info())
print(X_obj_encoded.info())


X_num.reset_index(inplace=True)
X_obj_encoded.reset_index(inplace=True)


X = pd.concat([X_num, X_obj_encoded],
              axis = 1)

print(X.info())


print(y.value_counts())
print(y.value_counts() / len(y))

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               stratify=y,
                                               random_state=0)


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

model = RandomForestClassifier(n_estimators=100,
                               max_depth=3,
                               max_samples=1.0,
                               class_weight='balanced',
                               n_jobs=-1,
                               random_state=0)

model.fit(X_train,y_train)

score = model.score(X_train, y_train)
print(f'Score(Train) : {score}') 
                            
score = model.score(X_test, y_test)
print(f'Score(Test) : {score}') 


























