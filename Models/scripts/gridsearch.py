import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.simplefilter('ignore')

# Hyperparameters to optimize

ridge_params = {
    'max_iter' : [4, 6, 8, 10],
    'alpha' : [0.75, 1.0, 1.25],
    'random_state' : [1]
}

mlpr_params = {
    'hidden_layer_sizes' : [(80,), (100,), (120,)],
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter' : [4, 6, 8, 10],
    'power_t' : [0.25, 0.5, 0.75],
    'random_state' : [1]
}

rfreg_params = {
    'n_estimators' : [50, 75, 100, 125, 150],
    'max_depth' : [2, 4, 6, 8, 10],
    'max_features' : ['sqrt', 'log2', 1.0],
    'random_state' : [1]
}

xgbreg_params = {
    'n_estimators' : [50, 75, 100, 125, 150],
    'max_depth' : [2, 4, 6, 8, 10],
    'max_leaves' : [10, 15, 20],
    'random_state' : [1]
}

def main():

    X_train = pd.read_csv('.\data_files\X_train.csv').drop('Unnamed: 0', axis=1)
    y_train = np.array(pd.read_csv('.\data_files\y_train.csv').drop('Unnamed: 0', axis=1)).reshape(-1,)

    # Bulding models and GS objects
    ridge = Ridge()
    mlpr = MLPRegressor()
    rfreg = RandomForestRegressor()
    xgbreg = XGBRegressor()

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    ridge_gs = GridSearchCV(estimator=ridge, param_grid=ridge_params, cv=kf, scoring='r2', verbose=2)
    mlpr_gs = GridSearchCV(estimator=mlpr, param_grid=mlpr_params, cv=kf, scoring='r2', verbose=2)
    rfreg_gs = GridSearchCV(estimator=rfreg, param_grid=rfreg_params, cv=kf, scoring='r2', verbose=2)
    xgbreg_gs = GridSearchCV(estimator=xgbreg, param_grid=xgbreg_params, cv=kf, scoring='r2', verbose=2)

    for gs in [ridge_gs, mlpr_gs, rfreg_gs, xgbreg_gs]:
        gs.fit(X_train, y_train)
    

    # Bulding and fitting best scoring models
    ridge = ridge_gs.best_estimator_
    mlpr = mlpr_gs.best_estimator_
    rfreg = rfreg_gs.best_estimator_
    xgbreg = xgbreg_gs.best_estimator_

    for model in [ridge, mlpr, rfreg, xgbreg]:
        model.fit(X_train, y_train)

    # Pickling optimized models
    with open(f'./models/pickled/optimized_ridge.pickle', 'wb') as file:
        pickle.dump(ridge, file)

    with open(f'./models/pickled/optimized_mlpregressor.pickle', 'wb') as file:
        pickle.dump(mlpr, file)

    with open(f'./models/pickled/optimized_randomforestregressor.pickle', 'wb') as file:
        pickle.dump(rfreg, file)

    with open(f'./models/pickled/optimized_xgbregressor.pickle', 'wb') as file:
        pickle.dump(xgbreg, file)


if __name__ == '__main__':
    main()
