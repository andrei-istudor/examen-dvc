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
# @click.argument('input_folder', type=click.Path(exists=False), required=0)
# @click.argument('output_folder', type=click.Path(exists=False), required=0)
def main(input_folder= './data/processed_data'):
    """ performs grid search for hyperparameter tuning using SVR.
    """
    logger = logging.getLogger(__name__)
    logger.info('performing paramenter search for SVR')

    # input_folder = click.prompt('Enter the directory path for the input data', type=click.Path(exists=True))

    X_path = f'{input_folder}/X_train_scaled.csv'
    y_path = f'{input_folder}/y_train.csv'

    process_data(X_path, y_path)

def process_data(X_path, y_path):
    X_train = pd.read_csv(X_path)
    y_train = pd.read_csv(y_path)

    svc = svm.SVR()
    parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.1, 0.5, 1, 2, 5, 10], 'gamma':[0.001, 0.1, 0.5, 1, 2, 3]}
    clf = GridSearchCV(svc, parameters, n_jobs=4)

    clf.fit(X_train, y_train.squeeze())

    clfs = clf.cv_results_['params']
    scores = clf.cv_results_['std_test_score']
    for c_param, score in zip(clfs, scores):
        print(f'{c_param} : {score}')

    print(f'Best parameters: {clf.best_params_}')
    with open('./models/best_params.pkl', 'wb') as f:
        pickle.dump(clf.best_params_, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()


