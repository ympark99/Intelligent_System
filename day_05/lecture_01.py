#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:06:34 2022

@author: youngmin
"""

import pandas as pd
pd.options.display.max_columns = 100
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X.info()
X.isnull().sum()
X.describe(include='all')

y.head()
y.value_counts()
y.value_counts() / len(y)

from sklearn.model_selection import train_test_split

splits =  train_test_split(X, y,
                           test_size=0.3,
                           random_state=10,
                           stratify=y)

X_train=splits[0]
X_test=splits[1]
y_train=splits[2]
y_test=splits[-1]

X_train.head()
X_test.head()

X_train.shape
X_test.shape

y_train.value_counts() / len(y_train)
# X_train,X_test,y_train,y_test=\
y_test.value_counts() / len(y_test)




from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2',
                           C=1.0,
                         #  class_weight='balanced',
                           class_weight={0:1000, 1:1},
                           solver='lbfgs',
                           max_iter=1000000,
                           n_jobs=-1,
                           random_state=5,
                           verbose=3)


model.fit(X_train, y_train)


model.score(X_train, y_train)

model.score(X_test, y_test)

print(f'coef_ : {model.coef_}')

    
print(f'intercept_ : {model.intercept_}')
    

proba = model.predict_proba(X_train[:5])
proba

pred = model.predict(X_train[:5])
pred


df = model.decision_function(X_train[-10:])
df


pred = model.predict(X_train[-10:])
pred

y_train[:5]


from sklearn.metrics import confusion_matrix

pred = model.predict(X_train)
cm = confusion_matrix(y_train, pred)
cm

y_train.value_counts()


from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score


pred = model.predict(X_train)
ps = precision_score(y_train, pred, pos_label = 0)

ps

rs = recall_score(y_train, pred, pos_label = 0)

rs

























