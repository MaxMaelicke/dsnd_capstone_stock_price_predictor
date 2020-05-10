# -*- coding: utf-8 -*-
# Dash App for Stock Price Predictor
# Udacity Data Scientist Nanodegree

import dash
import dash_core_components as dcc
import dash_html_components as html

import colorlover as cl
import datetime as dt
import flask
import os
import pandas as pd
import time

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css']

app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets
)

colorscale = cl.scales['9']['qual']['Paired']

# Import Stock datasets
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


if __name__ == '__main__':
    app.run_server(debug=True)