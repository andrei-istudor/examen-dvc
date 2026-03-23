import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from pathlib import Path
import click

#NOTE: I took over code from the Liora DVC excercises to get familiar with the way this is done there

# @click.command()
# @click.argument('input_folder', type=click.Path(exists=False), required=0)
# @click.argument('output_folder', type=click.Path(exists=False), required=0)
def main(input_folder= './data/processed_data', output_folder= './data/processed_data'):
    """ createa a scaler from the training data and uses it to scale train and test data
    saves the scaled data in output_folder, together with the scaler - for further usage
    """
    logger = logging.getLogger(__name__)
    logger.info('scaling input data')

    # input_folder = click.prompt('Enter the directory path for the input data', type=click.Path(exists=True))
    # output_folder = click.prompt('Enter the directory path for the output preprocessed data (e.g., data/preprocessed_data)', type=click.Path())

    X_train_path = f'{input_folder}/X_train.csv'
    X_test_path = f'{input_folder}/X_test.csv'

    X_train_scaled_path = f'{output_folder}/X_train_scaled.csv'
    X_test_scaled_path = f'{output_folder}/X_test_scaled.csv'

    process_data(X_train_path, X_train_scaled_path, X_test_path, X_test_scaled_path)

def process_data(X_train_filepath, X_train_scaled_filepath, X_test_filepath, X_test_scaled_filepath):

    X_train = pd.read_csv(X_train_filepath)

    # Normalization between 0 and 1, I didn't check the distribution of the data but it should do
    min_max_scaler = MinMaxScaler()

    # print(X_train.head())
    X_train[X_train.columns] = min_max_scaler.fit_transform(X_train[X_train.columns])
    print('Normalize between 0 and 1')
    # print(X_train.head())
    X_train.to_csv(X_train_scaled_filepath, index=False)

    #scale the test data with the same scaler as the train data
    X_test = pd.read_csv(X_test_filepath)
    X_test[X_test.columns] = min_max_scaler.transform(X_test[X_test.columns])
    X_test.to_csv(X_test_scaled_filepath, index=False)

    #save the scaler - not specified, but nice to have
    #not the nicest way to get the scaler name, but I'm trying to keep the structure from the Exercises
    min_max_scaler_path = f'{X_train_scaled_filepath.replace("X_train_scaled.csv", "min_max_scaler.pkl")}'
    joblib.dump(min_max_scaler, min_max_scaler_path)
    print(f'scaler saved in {min_max_scaler_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()

