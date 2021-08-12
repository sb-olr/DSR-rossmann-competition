# -*- coding: utf-8 -*-

import logging
import click
import os
import pandas as pd
import pickle
import json
import xgboost as xgb
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils.utils import metric
from src.data.make_dataset import clean_data
from src.features.build_features import impute_modes, impute_state_holiday

DATA_DIRECTORY = os.path.join(os.getcwd(), "data/")
MODEL_DIRECTORY = os.path.join(os.getcwd(), "models/")

@click.command()
@click.argument('input_filepath', default=DATA_DIRECTORY, type=click.Path(exists=True))
@click.argument('model_filepath', default=MODEL_DIRECTORY, type=click.Path(exists=True))
@click.argument('testfile', default="holdout.csv")
def click_main(input_filepath, model_filepath, testfile):
    """Interface for Click CLI."""
    main(input_filepath=input_filepath, model_filepath=model_filepath, testfile=testfile)


def main(input_filepath, model_filepath, testfile):
    """
    Run model prediction on test data set
    """
    logger = logging.getLogger(__name__)

    logger.info('Importing test data')
    test_df = pd.read_csv(os.path.join(input_filepath, testfile))

    logger.info('Merge test data with store data')
    store_df = pd.read_csv(os.path.join(input_filepath, 'store.csv'))
    data = test_df.merge(store_df, how="left", left_on='Store', right_on='Store')

    logger.info('Cleaning data')
    clean_df = clean_data(data)

    logger.info('Check and fill missing values')
    # hard coded modes for test set based on training set
    mode_open = 1.0
    mode_promo = 0.0
    mode_sh = 0.0
    clean_df["Open"].fillna(value=mode_open, inplace=True)
    clean_df["Promo"].fillna(value=mode_promo, inplace=True)
    clean_df["SchoolHoliday"].fillna(value=mode_sh, inplace=True)

    df = impute_state_holiday(clean_df)

    logger.info('mean encode categorical vars')
    df = test_mean_encoding(df, "StoreType", "Sales")
    df = test_mean_encoding(df, "Assortment", "Sales")
    df = test_mean_encoding(df, "Store", "Sales")

    y = df["Sales"]
    X = df.copy()

    logger.info('Features used in the model:')
    test_cols = ['Open', 'Promo', 'Month', 'Year', 'Weekday', 'SchoolHoliday',
                  'Holiday', 'StoreType_enc', 'Assortment_enc', 'Store_enc'
                  ]
    print(test_cols)
    X = X[test_cols]

    logger.info('load pre-trained model')
    model = pickle.load(open(os.path.join(model_filepath, "xgb_model.dat"), "rb"))

    logging.info('Make model predictions')
    y_pred = model.predict(X)
    result = metric(y_pred, y.to_numpy().flatten())

    logger.info(f'Model performance RMSPE: {result}%')


def test_mean_encoding(df: pd.DataFrame, col: str, on: str):
    with open(os.path.join(DATA_DIRECTORY, col+'_dict.json'), 'r') as fp:
        map_dict = json.load(fp)

    df[col+'_enc'] = df[col].map(map_dict).fillna(map_dict.get("NaN"))

    return df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    click_main()
