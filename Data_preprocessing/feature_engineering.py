import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
import scipy.stats

def main():
    '''
    Steps we are going to take in the script:
    1. Split data into train and test sets
    2. One hot encoding province, mark, fuel
    3. Encoding mark and generation
    4. Handling outliers, normalizing and scaling numerical features
    '''
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')

    logger.info('Reading data')
    data = pd.read_csv('.\Data\cleaned_data.csv').drop('Unnamed: 0', axis=1)

    # Step 1
    logger.info('Splitting data into train set and test set')
    X, y = data.drop('price', axis=1), data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Step 2
    not_polish_provinces = data['province'].value_counts().sort_values(ascending=False).index[16:] # 16 most popular values are polish provinces, we will replace the rest with value "other"
    X_train = X_train.replace(to_replace=not_polish_provinces, value='other')
    X_test = X_test.replace(to_replace=not_polish_provinces, value='other')
    
    features_to_encode = ['province','mark','fuel']

    logger.info('One hot encoding categorical columns')
    for col in features_to_encode:
        X_train = X_train.join(pd.get_dummies(X_train[col], prefix=col, dtype='float'))
        X_test = X_test.join(pd.get_dummies(X_test[col], prefix=col, dtype='float'))
    X_train = X_train.drop(features_to_encode, axis=1)
    X_test = X_test.drop(features_to_encode, axis=1)
    
    # Step 3
    features_to_looe_encode = ['model','generation_name']

    logger.info('Leave one out encoding "model" and "generation_name"')
    for col in features_to_looe_encode:
        looe = LeaveOneOutEncoder(cols=[col], sigma=0.1, random_state=1)
        X_train[col] = looe.fit_transform(X_train[col], y_train)
        X_test[col] = looe.transform(X_test[col])
    
    # Step 4
    features_to_process = ['model', 'generation_name', 'year', 'mileage', 'vol_engine']
    
    logger.info('Replacing outlietrs with winsorized mean')
    for col in features_to_process:
        IQR = scipy.stats.iqr(X_train[col])
        Q1 = np.percentile(X_train[col], 25)
        Q3 = np.percentile(X_train[col], 75)
        floor = Q1 - IQR
        ceil = Q3 + IQR
        min_limit = scipy.stats.percentileofscore(X_train[col], floor) / 100
        max_limit = scipy.stats.percentileofscore(X_train[col], ceil) / 100

        X_train[col] = scipy.stats.mstats.winsorize(X_train[col], limits=[min_limit, 1 - max_limit])
        X_test[col] = scipy.stats.mstats.winsorize(X_test[col], limits=[min_limit, 1 - max_limit])
    
    logger.info('Normalizing and scaling features')
    for col in features_to_process:
        pt = PowerTransformer()
        X_train[col] = pt.fit_transform(np.array(X_train[col]).reshape(-1, 1))
        X_test[col] = pt.transform(np.array(X_test[col]).reshape(-1, 1)) 
        
        scaler = MinMaxScaler()
        X_train[col] = scaler.fit_transform(np.array(X_train[col]).reshape(-1, 1))
        X_test[col] = scaler.transform(np.array(X_test[col]).reshape(-1, 1))
    
    logger.info('Checking data')
    logger.info('Train data shape: ')
    logger.info(X_train.shape)
    logger.info('Test data shape: ')
    logger.info(X_test.shape)
    logger.info('Sample of train data: ')
    logger.info(X_train.sample(10))
    logger.info('Sample of test data: ')
    logger.info(X_test.sample(10))
    logger.info('Description of train data: ')
    logger.info(X_train.describe())
    logger.info('Description of test data: ')
    logger.info(X_test.describe())

    logger.info('Exporting preprocessed data to csv files')
    X_train.to_csv('.\Data\X_train.csv')
    X_test.to_csv('.\Data\X_test.csv')
    y_train.to_csv('.\Data\y_train.csv')
    y_test.to_csv('.\Data\y_test.csv')


if __name__ == '__main__':
    main()
