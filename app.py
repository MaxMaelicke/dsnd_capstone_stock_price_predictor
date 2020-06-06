# -*- coding: utf-8 -*-
# Dash App for Stock Price Predictor
# Udacity Data Scientist Nanodegree
import os

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

#import plotly.graph_obj as go
#import plotly.io as pio

import datetime as dt
import flask
import time


# Styling
external_stylesheets = [dbc.themes.BOOTSTRAP]    # 'https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css']

#pio.templates.default = 'none'

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


# Import data
ticker_list = []
for filename in os.listdir('data'):
    if filename.endswith('.csv'):
        ticker_list.append(filename[:-4])


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
                {'label': ticker, 'value': ticker} for ticker in ticker_list
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
        dbc.Label('Show Error Histogram'),
        dbc.Col([
        daq.BooleanSwitch(
            id = 'histogram-switch',
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
                html.H3("Udacity's Data Scientist Nanodegree Capstone Project"),
                html.H3('FOR EDUCATIONAL PURPOSES ONLY - USE AT OWN RISK'),
                html.Hr()
            ])
        ])
    ]),
    dbc.Row([
        dbc.Col(controls, md = 2),
        dbc.Col([
            dcc.Graph(id = 'line_chart', config = {'modeBarButtonsToRemove': ['lasso2d']}),
            dcc.Graph(id = 'histogram', config = {'modeBarButtonsToRemove': ['lasso2d']})
            ], width = 'auto'
            )
        ], align = 'center')
    ], fluid = 'True'
    )



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
