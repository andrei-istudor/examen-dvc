import pandas as pd
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import pickle
import logging
from pathlib import Path
import click

#NOTE: I took over code from the Liora DVC excercises to get familiar with the way this is done there

# @click.command()
# @click.argument('data_folder', type=click.Path(exists=False), required=0)
# @click.argument('models_folder', type=click.Path(exists=False), required=0)
def main(data_folder= './data/processed_data', models_folder= './models'):
    """ trains the model with the best parameters found in the previous step.
    """
    logger = logging.getLogger(__name__)
    logger.info('training the model with the best parameters found in the previous step')

    # data_folder = click.prompt('Enter the directory path for the input data', type=click.Path(exists=True))
    # models_folder = click.prompt('Enter the directory path for the models', type=click.Path(exists=True))

    X_path = f'{data_folder}/X_train_scaled.csv'
    y_path = f'{data_folder}/y_train.csv'
    params_path = f'{models_folder}/best_params.pkl'

    process_data(X_path, y_path, params_path)

def process_data(X_path, y_path, params_path):
    X_train = pd.read_csv(X_path)
    y_train = pd.read_csv(y_path)

    svc = svm.SVR()

    with open(params_path, 'rb') as f:
        saved_params = pickle.load(f)

    new_model = svm.SVR(**saved_params)
    new_model.fit(X_train, y_train.squeeze())

    with open(params_path.replace('best_params.pkl', 'best_model.pkl'), 'wb') as f:
        pickle.dump(new_model, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()


