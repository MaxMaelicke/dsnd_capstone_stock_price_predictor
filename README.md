# Stock Price Predictor  /  Dash App
### Udacity Data Scientist Nanodegree | Capstone Project

This project inhabitates the code of my Dash app predicting future stock prices and showing the predictor's performance against real test data. Data exploration, engineering, model selection and tuning is documented in the corresponding Juypter Notebook. This is the Capstone Project of my Data Scientist Nanodegree at Udacity.

As predictors you can choose between:
* Multiple Linear Regression
* Polynomial Features Regression
* Support Vector Regression
* LASSO Regression
* Ridge Regression
* Elastic Net Regression

A live version of the app (with outdated data) is hosted at: (http://maelicke.net:5000)

**Disclaimer:** Do not use this project for investment decisions. This project is for education purposes only.


## Previews
### Dash App with prediction and graphs comparing past predictions with past actual data
![Dash App](/screenshots/app.png?raw=true "dash_app")

### Jupyter-notebook excerpts of model comparison
![Model comparison](/screenshots/jn_model_comparison.png?raw=true "model_comparison")  
![Line graphs](/screenshots/jn_line_graphs.png?raw=true "line_graphs")  
![Error histogram](/screenshots/jn_histogram.png?raw=true "error_histogram")


## Possible improvements
The app could be improved by integrating the automatic retrieval of up-to-date stock data. For performance issues data retrieval and model training could be done daily at night.  

The predictor / regression models could be improved by taking into consideration other data than only price data, e. g. market sentiment about specific stocks, developments of other markets like currencies or precious metals, and developments between specific market segments like automotive or tech (see Jupyter Notebook for more details).


## Instructions
Download or clone this repository.

Download stock data (Open, High, Low, Close, Adjusted Close, Volume; e. g. from Yahoo Finance) and save as csv file into folder 'data'.  

Run python script 'py_scripts/create_static_data.py' to create pickle  data for the app.

Run python script app.py.

Visit http://127.0.0.1:8050 to see the Dash app.  

*Optional - for production server:*  
Add function/script to automatically retrieve updated stock data.  
Setup automatic job to run 'create_static_data.py' when new stock data is available.  
Setup Gunicorn/WSGI sock (e.g. systemd service) to point to Dash app.  
Setup webserver and link to app location.  


## Prerequisites & Used libraries

* Python 3

* Numpy
* Pandas

* Dash (dash_core_components, dash_html_components, dash_bootstrap_components, dash_daq)
* Plotly
* Flask


* Scikit-Learn (train_test_split, GridSearchCV, MinMaxScaler, OneHotEncoder, PolynomialFeatures, StandardScaler, LinearRegression, Lasso, Ridge, ElasticNet, RandomForestRegressor, SVR, r2_score, mean_squared_error)

* Os
* Pickle
* Datetime
* Time


## Author

* **Max Maelicke** - (https://github.com/MaxMaelicke)


## Acknowledgements

Many thanks to
* **Udacity** for this Nanodegree Program and providing the project idea (https://www.udacity.com/),
* **Plotly / Dash** for their awesome open-source graphing and dashboard libraries (https://plotly.com/),
* **Yahoo Finance** for the stock price data (https://finance.yahoo.com/).
