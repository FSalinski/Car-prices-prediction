import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.simplefilter('ignore')

logging.basicConfig(level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')

models_dict = {
    'ridge' : Ridge(random_state=1),
    'mlpregressor' : MLPRegressor(random_state=1),
    'randomforestregressor' : RandomForestRegressor(max_depth=8, random_state=1),
    'xgbregressor' : XGBRegressor(random_state=1)
}

def train_and_pickle(model, X, y):
    logging.info(f'Training {model}')
    models_dict[model].fit(X, y)

    logging.info(f'Pickling {model}')
    with open(f'./models/pickled/base_{model}.pickle', 'wb') as file:
        pickle.dump(models_dict[model], file)

def main():
                    
    X_train = pd.read_csv('.\Data\X_train.csv').drop('Unnamed: 0', axis=1)
    y_train = pd.read_csv('.\Data\y_train.csv').drop('Unnamed: 0', axis=1)

    for model in models_dict:
        train_and_pickle(model, X_train, np.array(y_train).reshape(-1,))
    

if __name__ == '__main__':
    main()
