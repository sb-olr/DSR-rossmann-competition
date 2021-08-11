# -*- coding: utf-8 -*-

import logging
import click
import numpy as np
import os
import json
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from data.data_path import DATA_DIRECTORY
from sklearn.model_selection import train_test_split


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
    logger.info('Run build features scripts')

    logger.info('importing data')
    df_clean = pd.read_csv(os.path.join(input_filepath, 'df_clean.csv'))

    logger.info('Check and fill missing values')
    df = impute_modes(df_clean)
    df = impute_state_holiday(df)
    

    logger.info('train test split')
    # Get X and y
    y = df["Sales"]
    X = df.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    logger.info('mean encode categorical vars')
    X_train, X_test = apply_mean_encoding(X_train, X_test, "StoreType", "Sales")
    X_train, X_test = apply_mean_encoding(X_train, X_test, "Assortment", "Sales")
    X_train, X_test = apply_mean_encoding(X_train, X_test, "Store", "Sales")


    X_train.to_csv(os.path.join(output_filepath, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_filepath, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_filepath, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_filepath, 'y_test.csv'), index=False)

    logger.info(f"Split data files written to {output_filepath}")


def mean_encoding(df: pd.DataFrame, col: str, on: str):
    # mean_encoding(df, "StoreType", "Sales")
    overall_mean=df[on].mean()
    map_dict = df.groupby(col).mean().loc[:, on].to_dict()
    map_dict["NaN"] = overall_mean
    return map_dict


def apply_mean_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, on: str, savepath=DATA_DIRECTORY):
    #storetype_dict
    map_dict = mean_encoding(df_train, col, on)
    df_train[col+'_enc'] = df_train[col].map(map_dict).fillna(map_dict.get("NaN"))
    df_test[col+'_enc'] = df_test[col].map(map_dict).fillna(map_dict.get("NaN"))
    # os.path
    with open(os.path.join(savepath, col+'_dict.json'), 'w') as fp:
        json.dump(map_dict, fp)

    return df_train, df_test


def impute_modes(df: pd.DataFrame):
    mode_open=df["Open"].mode()[0]
    mode_promo=df["Promo"].mode()[0]
    mode_SH=df["SchoolHoliday"].mode()[0]

    df["Open"].fillna(value=mode_open, inplace=True)
    df["Promo"].fillna(value=mode_promo, inplace=True)
    df["SchoolHoliday"].fillna(value=mode_SH, inplace=True)

    return df


def impute_state_holiday(df: pd.DataFrame):
    # Create new feature Holiday, initially by default no holiday (0)
    df["Holiday"] = 0

    # Whenever StateHoliday indicates a holiday make it 1
    mask = (df["StateHoliday"].isin(["a", "b", "c"]))
    df.loc[mask, "Holiday"] = 1

    # Whenever StateHoliday is missing and store is closed, make it a holiday
    mask2 = ((df["StateHoliday"].isna()) & (df["Open"] == 0))
    df.loc[mask2, "Holiday"] = 1

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

