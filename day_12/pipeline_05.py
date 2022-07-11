stratify = y,
random_state=11)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

base_model = \
    LogisticRegression(n_jobs=-1, random_state=11)
    
pipe = Pipeline([
    ('s_scaler', scaler),
    ('base_model', base_model)])

param_grid = [{'base_model__penalty' : ['L2'],
               'base_model__solver' : ['Lbfgs'],}]