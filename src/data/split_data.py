from sklearn.model_selection import train_test_split
import pandas as pd

import logging
from pathlib import Path
import click

#NOTE: I took over code from the Liora DVC excercises to get familiar with the way this is done there

# @click.command()
# @click.argument('input_folder', type=click.Path(exists=False), required=0)
# @click.argument('output_folder', type=click.Path(exists=False), required=0)
def main(input_folder= './data/raw_data', output_folder= './data/processed_data'):
    """ 
    Data Splitting: 
    Split the data into training and testing sets. 
    Our target variable is silica_concentrate, located in the last column of the dataset. 
    This script will produce 4 datasets (X_test, X_train, y_test, y_train) 
    that you can store in data/processed*.
    *(the strucuture on the GitHub has data/raw_data and data/processed_data)

    Splits the raw data (from (../data/raw_data) in train and test sets 
    (saved in../data/preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('splitting the raw data into train and test sets + target')

    # input_folder = click.prompt('Enter the directory path for the input data', type=click.Path(exists=True))
    # output_folder = click.prompt('Enter the directory path for the output preprocessed data (e.g., data/preprocessed_data)', type=click.Path())

    input_filepath = f'{input_folder}/raw.csv'

    process_data(input_filepath, output_folder)

def process_data(input_filepath, output_folder):
    # Load your raw data from input_filepath
    raw_data = pd.read_csv(input_filepath)

    y = raw_data['silica_concentrate']
    #remove also the date, it's not a feature we want to use for the prediction
    X = raw_data.drop(columns=['silica_concentrate', 'date'])

    # print(X.head())
    # print(X.shape)  
    # print(y.shape)
    # print(y.head())


    # Perform your train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=222)

    # Save the train test split data to output_folder
    X_train.to_csv(f'{output_folder}/X_train.csv', index=False)
    X_test.to_csv(f'{output_folder}/X_test.csv', index=False)
    y_train.to_csv(f'{output_folder}/y_train.csv', index=False)
    y_test.to_csv(f'{output_folder}/y_test.csv', index=False)
    print(f'Train and test data have been saved to {output_folder}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()