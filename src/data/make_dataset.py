# -*- coding: utf-8 -*-

import logging
import click
import numpy as np
import os
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
#from data.data_path import DATA_DIRECTORY

DATA_DIRECTORY = os.path.join(os.getcwd(), "data/")
#DATA_DIRECTORY = os.path.join("../../", os.path.dirname(os.path.realpath(__file__)), 'data/')

@click.command()
@click.argument('input_filepath', default=DATA_DIRECTORY, type=click.Path(exists=True))
@click.argument('output_filepath', default=DATA_DIRECTORY, type=click.Path())
def click_main(input_filepath, output_filepath):
    """Interface for Click CLI."""
    main(input_filepath=input_filepath, output_filepath=output_filepath)


def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to download and merge data.
    """
    logger = logging.getLogger(__name__)
    logger.info('Make dataset from raw data')

    logger.info('Merge train data with store data')
    train_df = pd.read_csv(os.path.join(input_filepath, 'train.csv'))
    store_df = pd.read_csv(os.path.join(input_filepath, 'store.csv'))
    data = train_df.merge(store_df, how="left", left_on='Store', right_on='Store')

    logger.info('Cleaning data')
    clean_df = clean_data(data)
    clean_df.to_csv(os.path.join(output_filepath, 'df_clean.csv'), index=False)

    logger.info(f"Clean data written to {output_filepath}")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # drop all rows with missing and zero sales (otherwise error in metric)
    df = data.dropna(subset=["Sales"])
    df = df.loc[~(df["Sales"] == 0)]

    # drop all rows with missing store IDs
    df = df.dropna(subset=['Store'])

    # Get relevant date features
    var_date = 'Date'
    var_month = 'Month'
    var_year = 'Year'
    var_day = 'Weekday'

    df[var_date] = pd.to_datetime(df[var_date])
    df[var_month] = df[var_date].dt.month
    df[var_year] = df[var_date].dt.year
    df[var_day] = df[var_date].dt.dayofweek
    df.drop(['DayOfWeek'], axis=1, inplace=True)

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