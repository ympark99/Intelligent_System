import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)

y = pd.Series(data.target, name = 'target')

print(X.head())
print(X.info())
print(X.describe())

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3,
                                                 stratify=y,
                                                 random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0,class_weight='balanced', random_state=0)

model.fit(X_train, y_train)

v_score = model.score(X_train_scaled, y_train)
print(f'학습 : {v_score}')
v_score = model.score(X_test_scaled, y_test)
print(f'테스트 : {v_score}')

# 선형 모델은 각각의 특성들에 대해서 가중치를 계산하는 모델
print(f'학습된 가중치 : \n{model.coef_}')

print(X.info())