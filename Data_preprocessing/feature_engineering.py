import logging
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    '''
    Steps we are going to take in the script:
    1. Split data into train and test sets
    2. One hot encoding province, mark, fuel
    3. Encoding mark and generation
    4. Scaling, normalizing and dropping outliers in numerical features
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
    


if __name__ == '__main__':
    main()