# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import logging
import click
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from data.data_path import DATA_DIRECTORY
from src.utils.utils import metric


@click.command()
@click.argument('input_filepath', default=DATA_DIRECTORY, type=click.Path(exists=True))
@click.argument('output_filepath', default=DATA_DIRECTORY, type=click.Path())
def click_main(input_filepath, output_filepath):
    """Interface for Click CLI."""
    main(input_filepath=input_filepath, output_filepath=output_filepath)


def main(input_filepath, output_filepath):
    """
    Runs build features scripts.
    """
    logger = logging.getLogger(__name__)
    logger.info('Train models')

    logger.info('Importing train/test data')
    X_train = pd.read_csv(os.path.join(input_filepath, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(input_filepath, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(input_filepath, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(input_filepath, 'y_test.csv'))

    logger.info('Features used during training')
    train_cols = ['Open', 'Promo', 'Month', 'Year', 'Weekday', 'SchoolHoliday',
            'Holiday', 'StoreType_enc', 'Assortment_enc', 'Store_enc'
            ]
    print(train_cols)
    X_train = X_train[train_cols]
    X_test = X_test[train_cols]

    logger.info('Benchmark models with different measures')
    result_dict = train_benchmark(X_train, X_test, y_train, y_test)
    best_benchmark = min(result_dict, key=result_dict.get)
    logger.info(f'The best performing benchmark model is always predicting the: {best_benchmark}, '
                f'RMSPE: {round(result_dict[best_benchmark], 2)}%')

    # logger.info('Run random forest model')
    # result = random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_features=0.8, max_depth=8)
    # logger.info(f'The best performing random forest model has RMSPE: {result}%')

    logger.info('Run gradient boost model')
    result = gradient_booster(X_train, X_test, y_train, y_test, n_estimators=500, colsample_bytree= 0.8,
                                    eta=0.1, max_depth= 7, subsample= 0.7)

    logger.info(f'The best performing random forest model has RMSPE: {result}%')




def train_benchmark(X_train, X_test, y_train, y_test):
    result_dict = {}

    mean_sales = y_train.mean()
    median_sales = y_train.median()
    mode_sales = y_train.mode()

    y_hat_mean = np.full((len(y_test), 1), mean_sales)
    result_dict["Mean"] = metric(y_hat_mean, y_test.to_numpy())

    y_hat_median = np.full((len(y_test), 1), median_sales)
    result_dict["Median"] = metric(y_hat_median, y_test.to_numpy())

    y_hat_mode = np.full((len(y_test), 1), mode_sales)
    result_dict["Mode"] = metric(y_hat_mode, y_test.to_numpy())

    return result_dict


def random_forest(X_train, X_test, y_train, y_test, n_estimators:int, max_features, max_depth:int):
    rf_model = RandomForestRegressor(n_estimators=n_estimators,
                                     max_features=max_features,
                                     max_depth=max_depth,
                                     criterion="mse",
                                     random_state=42,
                                     n_jobs=-1)

    y_train = y_train.to_numpy().flatten()
    y_test = y_test.to_numpy().flatten()

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    result = metric(y_pred, y_test)

    return round(result, 2)

def gradient_booster(X_train, X_test, y_train, y_test, n_estimators:int, max_depth:int, subsample:float, colsample_bytree:float, eta:float, n_jobs=-1, random_state=42):
    xgboost_model=xgb.XGBRegressor(
                                n_estimators = n_estimators,
                                max_depth = max_depth,
                                subsample = subsample,
                                n_jobs= n_jobs,
                                random_state= random_state,
                                colsample_bytree = colsample_bytree,
                                eta = eta
                                )


    y_train = y_train.to_numpy().flatten()
    y_test = y_test.to_numpy().flatten()

    xgboost_model.fit(X_train, y_train)

    y_pred = xgboost_model.predict(X_test)
    result = metric(y_pred, y_test)

    return round(result, 2)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    click_main()
