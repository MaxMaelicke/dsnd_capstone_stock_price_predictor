# -*- coding: utf-8 -*-
# Dash App for Stock Price Predictor
# Udacity Data Scientist Nanodegree
import os
import pickle

import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta  

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
        

# Load pickel data per ticker, regressor and time window
# pickle data (machine-learning results) are created with py_scripts/create_static_data.py
def get_window_pred_data(ticker, regressor, window):
    '''
    Function to import data for app from pre-compiled static pickle data.
    Input:    ticker                  str; ticker of stock
              regressor               str; regressor (from regressor_scalers)
              window                  float; time window for predictions
    Output:   window_pred_data        list of arrays containing ticker, regressor, window, y_pred, y_test, err (error), err_perc (error in %), y_pred_last
    '''
    with open('data/pickles/' + ticker + '_' + regressor + '_' + str(window) + '.pkl', 'rb') as file:
            window_pred_data = pickle.load(file)
    
    return window_pred_data
    

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
        dbc.Label('Show Error Line'),
        dbc.Col([
        daq.BooleanSwitch(
            id = 'error-switch',
            on = True,
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
                html.H5('DO NOT USE FOR INVESTMENT DECISIONS - for educational purposes only', className='text-muted'),
                html.H5('created with Dash by Plotly', className='text-muted'),
                html.Hr()
            ])
        ])
    ]),
    dbc.Row([
        dbc.Col(controls, md = 2),
        dbc.Col([
                html.H4('Adjusted Close Predictions'),
                dcc.Tabs(id = 'window-tab', value = '1', children = [
                        dcc.Tab(label = '1-day window', value = '1', children = [
                                dbc.Card(
                                        dbc.CardBody([
                                                dbc.Row(html.H5(id = 'card-header')),
                                                dbc.Row([
                                                        dcc.Graph(id = 'line_chart1', config = {'modeBarButtonsToRemove': ['lasso2d']}),
                                                        dbc.Card(id = 'pred-card')
                                                        ]),
                                                dbc.Row(html.H6('Distribution of Errors in %')),
                                                dbc.Row(dcc.Graph(id = 'histogram'))
                                                    ])
                                        )
                                ]),
                                        
                        dcc.Tab(label = '5-day window', value = '5', children = [
                                dbc.Card(
                                        dbc.CardBody([
                                                dbc.Row(html.H5(id = 'card-header5')),
                                                dbc.Row([
                                                        dcc.Graph(id = 'line_chart5', config = {'modeBarButtonsToRemove': ['lasso2d']}),
                                                        dbc.Card(id = 'pred-card5')
                                                        ]),
                                                dbc.Row(html.H6('Distribution of Errors in %')),
                                                dbc.Row(dcc.Graph(id = 'histogram5'))
                                                    ])
                                        )
                                ]),
                        dcc.Tab(label = '10-day window', value = '10', children = [
                                dbc.Card(
                                        dbc.CardBody([
                                                dbc.Row(html.H5(id = 'card-header10')),
                                                dbc.Row([
                                                        dcc.Graph(id = 'line_chart10', config = {'modeBarButtonsToRemove': ['lasso2d']}),
                                                        dbc.Card(id = 'pred-card10')
                                                        ]),
                                                dbc.Row(html.H6('Distribution of Errors in %')),
                                                dbc.Row(dcc.Graph(id = 'histogram10'))
                                                    ])
                                        )
                                ]),
                        dcc.Tab(label = '20-day window', value = '20', children = [
                                dbc.Card(
                                        dbc.CardBody([
                                                dbc.Row(html.H5(id = 'card-header20')),
                                                dbc.Row([
                                                        dcc.Graph(id = 'line_chart20', config = {'modeBarButtonsToRemove': ['lasso2d']}),
                                                        dbc.Card(id = 'pred-card20')
                                                        ]),
                                                dbc.Row(html.H6('Distribution of Errors in %')),
                                                dbc.Row(dcc.Graph(id = 'histogram20'))
                                                    ])
                                        )
                                ]),
                        
                        ]),
                html.Div([
                            html.Hr(),
                            dcc.Markdown('''
                                         When evaluating the different models **Ridge Regression** performed best.  
                                         The methodology, model comparison and source code can be accessed on my [Github project](https://github.com/CSSEGISandData/COVID-19).
                                           
                                         This project was conducted as Capstone project for my Data Scientist Nanodegree at [Udacity](https://www.udacity.com/).  
                                         
                                         If you want to learn more about me, please check out my  
                                         * [Homepage](https://www.maelicke.net)
                                         * [Github](https://github.com/MaxMaelicke)
                                         * [LinkedIn](https://www.linkedin.com/in/max-maelicke-025a35137/)
                                         * [Xing](https://www.xing.com/profile/Max_Maelicke).
                                         ''')
                            ])
                ])
        ])
    ], fluid = 'True')


# Callbacks (Interactive inputs for graphs)
@app.callback(
        [# Output 1-day window
        Output('card-header', 'children'),
        Output('line_chart1', 'figure'),
        Output('pred-card', 'children'),
        Output('histogram', 'figure'),
        # Output 5-day window
        Output('card-header5', 'children'),
        Output('line_chart5', 'figure'),
        Output('pred-card5', 'children'),
        Output('histogram5', 'figure'),
        # Output 10-day window
        Output('card-header10', 'children'),
        Output('line_chart10', 'figure'),
        Output('pred-card10', 'children'),
        Output('histogram10', 'figure'),
        # Output 20-day window
        Output('card-header20', 'children'),
        Output('line_chart20', 'figure'),
        Output('pred-card20', 'children'),
        Output('histogram20', 'figure'),
        ],
        [
        Input('stock-dropdown', 'value'),
        Input('regressor-dropdown', 'value'),
        Input('error-switch', 'on'),
        Input('window-tab', 'value')
        ]
        )
    
# Graph Generation
def update_graph(stock_dropdown, regressor_dropdown, on, window):
    '''
    Inputs:  stock_dropdown        str; filtered stock ticker
             regressor_dropdown    str; filtered regressor
             histogram_switch      bool; show/not show histogram
    Outputs: line_chart            plotly figure for comparing actual vs. predicted prices
             histogram             plotly figure; showing prediction errors
             card_header           str; header for window-card
             card_info_header      str; header for prediction info card
             card_info             str; prediction information for card
    '''
    # Load & prepare data for graphing
    # window_pred_data columns = y_pred, y_test, y_pred_last, date_arr
    window_pred_data = get_window_pred_data(stock_dropdown, regressor_dropdown, window)
    
    # y_pred array structure is different for three regressors
    if regressor_dropdown in ['Support Vector Regression', 'LASSO Regression', 'Elastic Net Regression']:
        y_pred = window_pred_data[0].tolist()
        y_pred_last = window_pred_data[2].tolist()[0]
    else:
        y_pred = [element[0] for element in window_pred_data[0].tolist()]
        y_pred_last = window_pred_data[2].tolist()[0][0]
    
    y_test = [element[0] for element in window_pred_data[1].tolist()]
    x_dates = [str(element) for element in window_pred_data[3][0]]
    
    # Calculate date for future prediction (window is only working days)
    if int(window) > 1:
        window_delta = int(window) * 2
        window_delta = window_delta + 2 * (window_delta // 5)
    else:
        window_delta = int(window)
    future_pred_date = datetime.strptime(x_dates[-1][:10], '%Y-%m-%d') + timedelta(days=window_delta)
    future_pred_date = datetime.date(future_pred_date)
    
    zip_err = zip(y_pred, y_test)
    err = [pred - test for pred, test in zip_err]
    
    zip_err_p = zip(err, y_test)
    err_perc = [err / test for err, test in zip_err_p]
    
    #err_perc = err / y_test
    err_mean = sum(err) / len(err)
    err_min = min(err)
    err_max = max(err)
    err_5_range = np.count_nonzero((np.array(err_perc) <= 0.05) & (np.array(err_perc) >= -0.05)) / len(err_perc)
    
    # Line Chart
    line_chart = go.Figure()
    line_chart.add_trace(go.Scatter(
            x = x_dates,
            y = y_test,
            name = 'Actual',
            mode = 'lines',
            line = dict(color = 'green')))
    line_chart.add_trace(go.Scatter(
            x = x_dates,
            y = y_pred,
            name = 'Predicted',
            mode = 'lines',
            line = dict(color = 'orange')))        
           
    line_chart.update_layout(
            autosize = False,
            height = 600,
            width = 800)
    
    # Error switch
    if on == True:
        # Line Chart
        line_chart.add_trace(go.Scatter(
            x = x_dates,
            y = err,
            name = 'Error abs.',
            mode = 'lines',
            line = dict(color = 'red')))
        
    # Histogram
    histogram = go.Figure()
    histogram.add_trace(go.Histogram(
            x = err_perc,
            histnorm = 'probability'))
    histogram.update_layout(autosize = False,
        height = 600,
        width = 800)
    
    # Card Output
    card_header = str(stock_dropdown) + ' | ' + str(regressor_dropdown)
    card_info_header = 'Prediction for ' + str(future_pred_date)
    card_info_1 = 'Adj. Close = ' + '{:.2f}'.format(y_pred_last)
    card_info_2 = 'Average Error = ' + '{:.2f}'.format(err_mean)
    card_info_3 = 'Min Error = ' + '{:.2f}'.format(err_min)
    card_info_4 = 'Max Error = ' + '{:.2f}'.format(err_max)
    card_info_5 = 'Errors within +/- 5% = ' + '{:.2%}'.format(err_5_range)
    
    pred_card = dbc.CardBody([
            html.H5(card_info_header, className = 'card-title'),
            html.P(card_info_1, className = 'card-text'),
            html.Hr(),
            html.P(card_info_2, className = 'card-text'),
            html.P(card_info_3, className = 'card-text'),
            html.P(card_info_4, className = 'card-text'),
            html.P(card_info_5, className = 'card-text')
            ])
    
    # Output    1-day window                              5-day window                                                10-day window                                  20-day window
    return card_header, line_chart, pred_card, histogram, card_header, line_chart, pred_card, histogram, card_header, line_chart, pred_card, histogram, card_header, line_chart, pred_card, histogram


if __name__ == '__main__':
    app.run_server(debug=True)
