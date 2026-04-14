import pandas as pd
import pickle
import logging
from pathlib import Path
import click
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm

#NOTE: I took over code from the Liora DVC excercises to get familiar with the way this is done there

# @click.command()
# @click.argument('data_folder', type=click.Path(exists=False), required=0)
# @click.argument('models_folder', type=click.Path(exists=False), required=0)
def main(data_folder= './data/processed_data', models_folder= './models'):
    """ computes predictions and evaluation metrics using the best model found in the previous step.
    Finally, using the trained model, 
    - we will evaluate its performance 
    - make predictions. 
    At the end of this script, we will have 
    - a new dataset in data containing the predictions, 
    - along with a scores.json file in the metrics directory that will capture evaluation metrics of our model (e.g., MSE, R2).
    """
    logger = logging.getLogger(__name__)
    logger.info('computing predictions and evaluation metrics')

    # data_folder = click.prompt('Enter the directory path for the input data', type=click.Path(exists=True))
    # models_folder = click.prompt('Enter the directory path for the models', type=click.Path(exists=True))

    X_path = f'{data_folder}/X_test_scaled.csv'
    y_path = f'{data_folder}/y_test.csv'
    models_path = f'{models_folder}/best_model.pkl'

    process_data(X_path, y_path, data_folder, models_path)

def process_data(X_path, y_path, data_folder, models_path):
    model = pickle.load(open(models_path, 'rb'))
    X_test = pd.read_csv(X_path)
    y_test = pd.read_csv(y_path)

    #compute the predictions
    y_pred = model.predict(X_test)

    #compute metrics and save them in a json file
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'mse': mse,
        'r2': r2
    }
    with open('./metrics/scores.json', 'w') as f:
        json.dump(metrics, f)

    y_pred_df = pd.DataFrame(y_pred, columns=['predictions'])
    y_pred_df.to_csv(f'{data_folder}/predictions.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()


