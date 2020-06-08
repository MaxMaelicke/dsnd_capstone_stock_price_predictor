#### Stock Price Predictor
#### Create static data for Dash App

# Import libraries
import os
import pickle

import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

from ml_scripts import import_engineer_data, split_scale_train_predict, import_data, preprocessing_data

# Regressor & Scaler Combinations
# SVR, LASSO, Ridge and ElasticNet work better with StandardScaler
# including GridSearch cross-validation for SVR, LASSO, Ridge and ElasticNet
svr_params = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
lasso_params = {'alpha': [1e-3, 1e-2, 1, 2, 5, 10]}
ridge_params = {'alpha': [1e-3, 1e-2, 1, 2, 5, 10]}
elastic_params = {'alpha': [1e-3, 1e-2, 1, 2, 5, 10], 'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}

regressors_scalers = {
    'Multiple Linear Regression': [LinearRegression(), MinMaxScaler(feature_range = (0, 1))],
    'Polynomial Features Regression': [LinearRegression(PolynomialFeatures(degree = 2)), MinMaxScaler(feature_range = (0, 1))],
    'Support Vector Regression': [GridSearchCV(SVR(), param_grid = svr_params, cv=10,n_jobs=-1), StandardScaler()],
    'LASSO Regression': [GridSearchCV(Lasso(), param_grid = lasso_params, cv=10,n_jobs=-1), StandardScaler()],
    'Ridge Regression': [GridSearchCV(Ridge(), param_grid = ridge_params, cv=10,n_jobs=-1), StandardScaler()],
    'Elastic Net Regression': [GridSearchCV(ElasticNet(), param_grid = elastic_params, cv=10,n_jobs=-1), StandardScaler()]
}

# Create ticker list
ticker_filenames = {}
for filename in os.listdir('../data'):
    if filename.endswith('.csv'):
        ticker_filenames[filename[:-4]] = '../data/' + filename

# Defining prediction windows in working days
# --> Date + 7 days = Date + 5 working days (since data dows not include weekends)
windows = [1, 5, 10, 20]

# Create static prediction data for each regressor, stock and window
for ticker in ticker_filenames:
    window_pred_data = []
    for regressor in regressors_scalers:
        for window in windows:
            X, y, X_last, date_arr = import_engineer_data(folder_path = ticker_filenames[ticker], window = window)
            y_pred, y_test, err, err_perc, mse, r2, adj_r2, y_pred_last = split_scale_train_predict(X, y, X_last, regressor = regressors_scalers[regressor][0], scaler = regressors_scalers[regressor][1])
            window_pred_data.append([[ticker, regressor, window, y_pred, y_test, err, err_perc, y_pred_last, date_arr]])
    # Save to file
    with open('../data/pickles/' + ticker + '.pkl', 'wb') as file:
        pickle.dump(window_pred_data, file)
