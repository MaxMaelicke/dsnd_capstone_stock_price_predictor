# -*- coding: utf-8 -*-
# Dash App for Stock Price Predictor
# Udacity Data Scientist Nanodegree
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

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output

import plotly.graph_objects as go
import plotly.io as pio

from py_scripts.ml_scripts import import_data, preprocessing_data, import_engineer_data, split_scale_train_predict

import datetime as dt
import flask
import time


# Styling
external_stylesheets = [dbc.themes.BOOTSTRAP]    # 'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css']

pio.templates.default = 'none'

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets
)


# Set prediction windows in working days
# --> Date + 7 days = Date + 5 working days (since data dows not include weekends)
windows = [1, 5, 10, 20]


# Define possible regressors and scaler combinations
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
for filename in os.listdir('data'):
    if filename.endswith('.csv'):
        ticker_filenames[filename[:-4]] = 'data/' + filename
        

# Create prediction for per ticker and time window
# Option 1: Import pre-compiled pickle data
def get_window_pred_data(ticker):
    '''
    Function to import data for app from pre-compiled static pickle data.
    Input:    ticker                  str; ticker of stock
    Output:   window_pred_data        list of arrays containing ticker, regressor, window, y_pred, y_test, err (error), err_perc (error in %), y_pred_last
              window_pred_data_ind    dictionary of indexes for regressors in window_pred_data
    '''
    with open('data/pickles/' + ticker + '.pkl', 'rb') as file:
        window_pred_data = pickle.load(file)
    
    window_pred_data_ind = {}
    for i in range(len(window_pred_data)):
        if i > 0:
            if window_pred_data[i][0][1] != window_pred_data[i-1][0][1]:
                window_pred_data_ind[window_pred_data[i][0][1]] = i
            else:
                continue
        else:
            window_pred_data_ind[window_pred_data[i][0][1]] = i
    
    return window_pred_data, window_pred_data_ind

# Option 2: Run and optimize model on data
# Create static prediction data for each regressor, stock and window
#def get_window_pred_data(ticker, regressor_scalers, windows):
#    '''
#    Function to create window_pred_data which consists of arrays of actual (test) prices, predicted prices and errors. This function calls other functions for
#    importing and preprocessing data,
#    training and optimizing models,
#    predicting with the appropriate model.
#    Input:  ticker                  str; ticker of stock
#            regressor               str; model regressor
#            regressor_scalers       dict; matching regressors with scaler and GridSearchCV parameters
#    Output: window_pred_data        list of arrays containing ticker, regressor, window, y_pred, y_test, err (error), err_perc (error in %), y_pred_last
#            window_pred_data_ind    dictionary of indexes for regressors in window_pred_data
#    '''
#    window_pred_data = []
#    for window in windows:
#            X, y, X_last = import_engineer_data(folder_path = ticker_filenames[ticker], window = window)
#            y_pred, y_test, err, err_perc, mse, r2, adj_r2, y_pred_last = split_scale_train_predict(X, y, X_last, regressor = regressor, scaler = regressors_scalers[regressor][1])
#            window_pred_data.append([[ticker, regressor, window, y_pred, y_test, err, err_perc, y_pred_last]])
#    window_pred_data_ind = {}
#    for i in range(len(window_pred_data)):
#    if i > 0:
#        if window_pred_data[i][0][1] != window_pred_data[i-1][0][1]:
#            window_pred_data_ind[window_pred_data[i][0][1]] = i
#        else:
#            continue
#    else:
#        window_pred_data_ind[window_pred_data[i][0][1]] = i
#    
#    return window_pred_data, window_pred_data_ind


# Create site layout and graphs
# Filters
controls = dbc.Card(
    [
    dbc.FormGroup(
        [
        dbc.Label('Stock'),
        dcc.Dropdown(
            id='stock-dropdown',
            options = [
                {'label': ticker, 'value': ticker} for ticker in ticker_filenames
            ],
            value = 'SPY'
        )
        ]
    ),
    dbc.FormGroup(
        [
        dbc.Label('Prediction Model'),
        dcc.Dropdown(
            id='regressor-dropdown',
            options = [
                {'label': regressor, 'value': regressor} for regressor in regressors_scalers
            ],
            value = 'Ridge Regression'
        )
        ]
    ),
    dbc.FormGroup(
        [
        dbc.Label('Show Error Information'),
        dbc.Col([
        daq.BooleanSwitch(
            id = 'error-switch',
            on = 'True',
            color = 'lightgreen'),
            ], md = 4,)
        ]
        )
        ]
    )

# Site layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1('Stock Price Predictor'),
                html.H5("Udacity's Data Scientist Nanodegree Capstone Project"),
                html.H5('DO NOT USE FOR INVESTMENT DECISIONS - for educational purposes only', className='text-muted'),
                html.H5('created with Dash by Plotly', className='text-muted'),
                html.Hr()
            ])
        ])
    ]),
    dbc.Row([
        dbc.Col(controls, md = 2),
        dbc.Col([
            dcc.Graph(id = 'line_chart')#, config = {'modeBarButtonsToRemove': ['lasso2d']}),
            #dcc.Graph(id = 'histogram', config = {'modeBarButtonsToRemove': ['lasso2d']})
            ], width = 'auto'
            )
        ], align = 'center')
    ], fluid = 'True'
    )


# Callbacks (Interactive inputs for graphs)
@app.callback([
        Output('line_chart', 'figure'),
        #Output('histogram', 'figure')
        ], [
        Input('stock-dropdown', 'value'),
        Input('regressor-dropdown', 'value'),
        Input('error-switch', 'on')
        ]
        )


# Graph Generation
def update_graph(stock_dropdown, regressor_dropdown, on):
    '''
    Inputs:  stock_dropdown        str; filtered stock ticker
             regressor_dropdown    str; filtered regressor
             histogram_switch      bool; show/not show histogram
    Outputs: line_chart            plotly figure for comparing actual vs. predicted prices
             forecast_card         dash card showing predicted price for window
             histogram             plotly figure showing prediction errors
             error_card            dash card showing metrics about prediction errors
    '''
    # Filter data
    # window_pred_data columns = ticker, regressor, window, y_pred, y_test, err, err_perc, y_pred_last, date_arr
    window_pred_data, window_pred_data_ind = get_window_pred_data(stock_dropdown)
    window = 0
    idx = window_pred_data_ind[regressor_dropdown]
    
    y_test = window_pred_data[idx][0][4].tolist()
    y_test = [element[0] for element in y_test]
    
    y_pred = window_pred_data[idx][0][3].tolist()
    y_pred = [element[0] for element in y_pred]
    
    x_dates = window_pred_data[idx][0][8].tolist()
    
    err = window_pred_data[idx][0][5].tolist()
    err = [element[0] for element in err]
    
    # Line Chart
    line_chart = go.Figure()
    line_chart.add_trace(go.Scatter(
            x = x_dates,
            y = y_test,
            name = 'Actual Adj. Close Price',
            mode = 'lines',
            line = dict(color = 'green')))
    
    line_chart.add_trace(go.Scatter(
            x = x_dates,
            y = y_pred,
            name = 'Predicted Adj. Close Price',
            mode = 'lines',
            line = dict(color = 'orange')))
    
    # Error-switch
    if on == True:
        line_chart.add_trace(go.Scatter(
                x = x_dates,
                y = err,
                name = 'Error abs.',
                mode = 'lines',
                line = dict(color = 'red')))
    
    #line_chart.show()
    
    return [line_chart]
'''
# Import stock datasets
df_MSFT = pd.read_csv('data/MSFT_5Y.csv')
df_MSFT['Stock'] = 'MSFT'

df_GOOG = pd.read_csv('data/GOOG_5Y.csv')
df_GOOG['Stock'] = 'GOOG'

df_AAPL = pd.read_csv('data/AAPL_5Y.csv')
df_AAPL['Stock'] = 'AAPL'

df_SPY = pd.read_csv('data/SPY_5Y.csv')
df_SPY['Stock'] = 'SPY'

# Merge datasets
frames = [df_MSFT, df_GOOG, df_AAPL, df_SPY]
df = pd.concat(frames)

app.layout = html.Div(className='col-12',
             children=[
                     html.Div(id='middle-info',
                              className='mt-3',
                              children=[
                                      html.H3('Stock Price Predictor'),
                                      html.H5('- DO NOT USE FOR INVESTMENT DECISIONS - for educational purposes only', className='text-muted'),
                                      html.H5('created with Dash by Plotly', className='text-muted'),
                                      ]
                              ),

    dcc.Dropdown(
        id='stock-ticker-input',
        options=[{'label': s[0], 'value': str(s[1])}
                 for s in zip(df.Stock.unique(), df.Stock.unique())],
        #value=['YHOO', 'GOOGL', 'MSFT'],
        multi=True
    ),

    # Option 1 - 1 graph per row
    #html.Div(id='graphs')

    # Option 2 - 2 graphs per row
    html.Div(#id='graphs',
             className='row mb-6',
             children=[
                     html.Div(id='graphs',
                              className='col-6')
                     ]
             )

], #className="container")
)

def bbands(price, window_size=10, num_of_std=5):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band

@app.callback(
    dash.dependencies.Output('graphs','children'),
    [dash.dependencies.Input('stock-ticker-input', 'value')])
def update_graph(tickers):
    graphs = []

    if not tickers:
        graphs.append(html.H3(
            "Select a stock ticker.",
            style={'marginTop': 20, 'marginBottom': 20}
        ))
    else:
        for i, ticker in enumerate(tickers):

            dff = df[df['Stock'] == ticker]

            candlestick = {
                'x': dff['Date'],
                'open': dff['Open'],
                'high': dff['High'],
                'low': dff['Low'],
                'close': dff['Close'],
                'type': 'candlestick',
                'name': ticker,
                'legendgroup': ticker,
                'increasing': {'line': {'color': colorscale[0]}},
                'decreasing': {'line': {'color': colorscale[1]}}
            }
            bb_bands = bbands(dff.Close)
            bollinger_traces = [{
                'x': dff['Date'], 'y': y,
                'type': 'scatter', 'mode': 'lines',
                'line': {'width': 1, 'color': colorscale[(i*2) % len(colorscale)]},
                'hoverinfo': 'none',
                'legendgroup': ticker,
                'showlegend': True if i == 0 else False,
                'name': '{} - bollinger bands'.format(ticker)
            } for i, y in enumerate(bb_bands)]
            graphs.append(dcc.Graph(
                id=ticker,
                figure={
                    'data': [candlestick] + bollinger_traces,
                    'layout': {
                        'margin': {'b': 0, 'r': 10, 'l': 60, 't': 0},
                        'legend': {'x': 0}
                    }
                }
            ))

    return graphs
'''

if __name__ == '__main__':
    app.run_server(debug=True)
