#### Stock Price Predictor
#### Import scripts

# Import libraries
import os

import pandas as pd
import numpy as np
import datetime


# Data import functions
def import_data(data = 'data/MSFT.csv'):
    '''
    Function to import and preprocess stock price data from Yahoo Finance for further usage.
    Input: Path to CSV file provided by Yahoo Finance
    Output: Preprocessed DataFrame
    '''
    df = pd.read_csv(data)
    df = preprocessing_data(df)
    return df


def preprocessing_data(df):
    '''
    Function to preprocess data: formatting, cleaning, forward-fill and backward-fill for nan values
    Input: df with stock price data
    Output: Preprocessed DataFrame
    '''
    # Format Date column
    df['Date'] = df['Date'].astype('datetime64[ns]')

    # Check if there are nan values
    for col in df.columns:
        if df[col].isnull().sum() >> 0:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        else:
            continue

    return df


def batch_import(folder_path = 'data'):
    '''
    Function to import and preprocess stock price data of several stocks from Yahoo Finance for further usage.
    Input: Path to Folder, which contains multiple CSV files provided by Yahoo Finance
    Output: Preprocessed DataFrame
    '''
    # Create list with file names of .csv files
    files = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            files.append(file)

    # Create first DataFrame
    df = import_data(os.path.join(folder_path, files[0]))
    ticker = files[0][:-4]
    df['Ticker'] = ticker

    # Find youngest first date of all stocks (we want to start the time series of each stock from the same date)
    startdate = df['Date'].min()
    for file in files[1:]:
        tmp_filename = os.path.join(folder_path, file)
        tmp_df = import_data(tmp_filename)
        tmp_date = tmp_df['Date'].min()
        if tmp_date > startdate:
            startdate = tmp_date

    # Filter df for equal timerange
    df = df[df['Date'] > startdate]

    # Merge other stock datasets
    for file in files[1:]:
        ticker = file[:-4]
        tmp_filename = os.path.join(folder_path, file)
        tmp_df = import_data(tmp_filename)
        tmp_df = tmp_df[tmp_df['Date'] > startdate]
        tmp_df['Ticker'] = ticker
        df = pd.concat([df, tmp_df])

    return df
