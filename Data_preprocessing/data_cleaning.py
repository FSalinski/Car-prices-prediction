import logging
import pandas as pd

def main():
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')

    logger.info('Reading data')
    data = pd.read_csv('.\Data\Car_Prices_Poland_Kaggle.csv').drop('Unnamed: 0', axis=1)

    logger.info('Dropping "city" column')
    data = data.drop('city', axis=1)

    logger.info('Filling null values in "generation_name" column')
    data['generation_name'] = data['generation_name'].fillna(value='non-specified')

    logger.info('Dropping record with mistake in "province"')
    record_to_drop = data[data['province'] == '('].index
    data = data.drop(index=record_to_drop, axis=0)

    logger.info('Checking data')
    logger.info('Data shape: ')
    logger.info(data.shape)
    logger.info('Null values: ')
    logger.info(data.isnull().sum())
    logger.info('Sample of data: ')
    logger.info(data.sample(10))

    logger.info('Exporting cleaned data to csv file')
    data.to_csv('.\Data\cleaned_data.csv')


if __name__ == '__main__':
    main()
