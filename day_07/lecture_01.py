import pandas as pd
pd.options.display.max_columns=100
pd.options.display.max_rows=10

fname_input = './titanic.csv'
data = pd.read_csv(fname_input,
                   header='infer',
                   sep=',')

data2 = data.drop(
    columns=['PassengerId','Name','Ticket','Cabin'],
    inplace=False)

data3 = data2.dropna(
    subset=['Age','Embarked'])

X = data3.iloc[:, 1:]
y = data3.Survived

# 수치형 데이터의 컬럼명
X_num = [cname for cname in X.columns 
         if X[cname].dtype in ['int64','float64'] ]
# 문자형 데이터의 컬럼명
X_obj = [cname for cname in X.columns 
         if X[cname].dtype not in ['int64','float64'] ]

X_num = X[X_num]
X_obj = X[X_obj]

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False,
                        handle_unknown='ignore')

X_obj_encoded = encoder.fit_transform(X_obj)

X_obj_encoded = pd.DataFrame(X_obj_encoded,
                             columns=['s_f','s_m',
                                      'e_C','e_Q','e_S'])

X_num.reset_index(inplace=True)
X_obj_encoded.reset_index(inplace=True)

# concat 메소드를 사용하여 데이터프레임을 결합
X = pd.concat([X_num, X_obj_encoded],
              axis = 1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               stratify=y,
                                               random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

model = RandomForestClassifier(n_estimators=100,
                               max_depth=10,
                               max_samples=1.0,
                               class_weight='balanced',
                               n_jobs=-1,
                               random_state=0)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'Score(Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score(Test) : {score}')


model2 = GradientBoostingClassifier(n_estimators=3,
                               max_depth=5,
                               random_state=0)

model2.fit(X_train, y_train)

score = model2.score(X_train, y_train)
print(f'Score(Train) : {score}')

score = model2.score(X_test, y_test)
print(f'Score(Test) : {score}')






















