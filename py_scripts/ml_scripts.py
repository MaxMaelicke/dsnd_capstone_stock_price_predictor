#### Stock Price Predictor
#### Machine Learning functions

# Import libraries
import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error


# Machine Learning functions
def import_engineer_data(batch_import = False, folder_path = 'data/SPY.csv', weekday_feature = False, window = 1):
    '''
    Function to import, preprocess and feature engineer stock price data of several stocks from Yahoo Finance for further usage.
    Inputs:
        batch_import       BOOL; True: Import all .csv files in given folder; False: Import single .csv file
        folder_path        STR; Path to Folder, which contains multiple CSV files provided by Yahoo Finance
                                If batch_import = False: folder_path must point to specific file, e.g. 'data/MSFT.csv'
        weekday_feature    BOOL; if true dummy variables for weekdays monday-friday are engineered
        window             INT; how many working days shall y be shifted from X (e. g. predict the price of tomorrow (1) or in one week (5))
    Outputs:
        X                  numpy array; independent variables
        y                  numpy array; dependent variable
        X_cols             list; showing column names, representing the names of the independent variables
    '''
    if batch_import:
        df = batch_import(folder_path)
        df = df.drop(['Ticker'], axis = 1)
    else:
        df = import_data(folder_path)

    if weekday_feature:
        # Create column with weekdays from Date column
        df['Date'] = df['Date'].astype('datetime64[ns]')
        df['Weekday'] = df['Date'].dt.day_name()

        # Create dummy variable columns for weekdays
        df = pd.concat([df, pd.get_dummies(df['Weekday'], drop_first = True)], axis = 1)

        # Define X
        X = df.drop(['Date', 'Weekday'], axis = 1).values

    else:
        X = df.drop(['Date'], axis = 1).values

    # Define y
    y = df.iloc[:, 5:6].values    # Adj. Close column

    # Windowing: Shift values to emulate y after 1, 5, 10 and 20 working days
    # --> Delete window rows at the beginning of y
    for i in range(window):
        y = np.delete(y, 0, 0)
    # --> Delete window rows at the end of X
        X = np.delete(X, -1, 0)

    return X, y


def split_scale_train_predict(X, y, regressor, scaler):
    '''
    Function to import, preprocess and feature engineer stock price data of several stocks from Yahoo Finance for further usage.
    Inputs:
        X              numpy array; independent variables
        y              numpy array; dependent variable
        X_cols         list; showing column names, representing the names of the independent variables
        regressor      regressor/model to be fitted and trained
    Output:
        y_pred         numpy array; predicted prices
    '''
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

    # Data Normalization
    scaler_X = scaler
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = scaler
    y_train = scaler_y.fit_transform(y_train.reshape(-1,1))

    # Fit Multiple Linear Regression Model to training set
    regressor = regressor
    regressor.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = regressor.predict(X_test)

    # Compare real adj. close with predicted value
    # Inverse normalization of y_pred to obtain interpretable values
    y_pred = scaler_y.inverse_transform(y_pred)

    # Calculate evaluation metrics
    err = y_test - y_pred
    err_perc = err / y_test
    err_5_range = np.count_nonzero((err_perc <= 0.05) & (err_perc >= -0.05)) / err_perc.size
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)     # sample size
    p = X.shape[1]    # number of explanatory variables
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Print model evaluation metrics
    print('--- Model Evaluation Metrics:')
    print('Mean squared error: {}'.format(mse))
    print('R-squared: {}'.format(r2))
    print('Adjusted R-squared: {}'.format(adj_r2))
    print('Mean error in %: {:.2%}'.format(err_perc.mean()))
    print('Error range: {:.2%} - {:.2%}'.format(err_perc.min(), err_perc.max()))
    print('Predictions within error range of -5% to 5%: {:.2%}'.format(err_5_range))

    return y_pred, y_test, err, err_perc, mse, r2, adj_r2
