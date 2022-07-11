import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=1,
    stratify=y)


param_grid = {'Learning_rate' : [0.1,0.2,0.3,1.,0.01],
              'max_depth' : [1,2,3],
              'n_estimators:':[100,200,300,10,50]}

# 교차 검증 점수를 기반으로 최적의 하이퍼 파라메터
from sklearn.model_selection import GridSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=1)


base_model = GradientBoostingClassifier(random_state=1)

grid_model = GridSearchCV(estimator = base_model, 
                          param_grid = param_grid,
                          cv=cv,
                          n_jobs=-1,
                          verbose=3)

grid_model.fit(X_train, y_train)


# 모든 하이퍼 파라메터를 조합하여 평가한
# 가장 높은 교차검증 score 값을 반환

print(f'best_score ->  {grid_model.best_score_}')

print(f'best_score ->  {grid_model.best_params_}')

print(f'best_score ->  {grid_model.best_estimator_}')

score = grid_model.score(X_train, y_train)
print(f'SCORE(TRAIN) : {score:.5f}')
score = grid_model.score(X_test, y_test)
print(f'SCORE(TEST) : {score:.5f}')