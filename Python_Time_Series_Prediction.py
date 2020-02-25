# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:22:32 2020

@author: bill-

#notice 1/3
installation fbprophet
-update conda
-install c++ compiler
-make sure dependencies are installed
-install pystan in conda-forge
-update index in the env
download arimas wheel file first and then install from it
"""
#load required packages
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import lightgbm as lgb
#from fbprophet import Prophet
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
#%%
#set pseudorandom number
seed = 7
np.random.seed(seed)
#%%
#PREDICTION OF VOLUME WEIGHTED AVERAGE PRICE
#converting the link
link = r"C:\Users\bill-\OneDrive\Dokumente\Docs Bill\TA_files\Trading_data_AI\msft_merged_May_June.csv"
link_1 = link.replace(os.sep, '/')
file = ''.join(link_1)
#%%
#loading the data set
df = pd.read_csv(file)
#set the date column as index to maneuver through the data more easily
df.set_index("date", drop = False, inplace = True)
#print the head
df.head(n = 3)
'''
Column names of given data frame; based on trading data from alpha vantage
pick or rename columns accordingly
Index(['date', '1. open', '2. high', '3. low', '4. close', '5. volume'], dtype = 'object')
'''
#%%
#plotting the vwap
#FIX; define function VWAP
#df.VWAP.plot(figsize = (14, 7))
#%%
#FEATURE ENGINEERING
#this is done to enhance the models prediction ability and accuracy
#typical engineered features based on lagging metrics
#mean + stdev of past 3d/7d/30d/ + rolling volume

df.reset_index(drop = True, inplace = True)
#pick lag features to iterate through and calculate features
#original lag features; based on tutorial dataset
#lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
lag_features = ["1. open", "2. high", "3. low", "4. close", "5. volume"]
t1 = 3
t2 = 7
t3 = 30

#rolling values for all columns ready to be processed
#rolling method; window = size of the moving window;
                #min_periods = min no. of obersvations that need to have a value(otherwise result is NA)
                #center = set labels at the center of the window
                #win_type = weighting of points, "None" all points are equally weighted
                #on = use datetime-like column index (instead of df indices) to calculate the value
                #axis = 0:row-wise; 1:column-wise
                #closed = close of
#DataFrame.rolling(self, window, min_periods = None, center = False, win_type = None, on = None, axis = 0, closed = None)
#DataFrame.shift(self, periods = 1, freq = None, axis = 0, fill_value = None)
                #periods = pos/ neg downwards or upwards shift in periods
                #freq = offset/timedelta/str; index shifted but data not realigned; extend index when shifting + preserve original data
                #axis = shift direction (0: index 1: columns None)
                #fill_value = numeric: np.nan; datetime,timedelta: NaT; extension types:dtype.na_value
df_rolled_3d = df[lag_features].rolling(window = t1, min_periods = 0)
df_rolled_7d = df[lag_features].rolling(window = t2, min_periods = 0)
df_rolled_30d = df[lag_features].rolling(window = t3, min_periods = 0)

#calculate the mean with a shifting time window
df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

#calculate the std dev with a shifting time window
df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df[f"{feature}_mean_lag{t1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{t2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{t3}"] = df_mean_30d[feature]

    df[f"{feature}_std_lag{t1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{t2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{t3}"] = df_std_30d[feature]

#fill missing values with the mean to keep distortion very low and allow prediction
df.fillna(df.mean(), inplace = True)
#associate date as the index columns to columns (especially the newly generated ones to allow navigating and slicing)
df.set_index("date", drop = False, inplace = True)
#%%
#add daytime features to add additional info the model can be fed
from datetime import datetime as dt
for date in df['date']:
    df.date = dt.strptime(date, '%Y-%m-%d; HH:MM:SS')
#df.date = pd.to_datetime(df.date, format = "%Y-%m-%d; %hh:%mm:%ms")
df["month"] = df.date.dt.month
df["week"] = df.date.dt.week
df["day"] = df.date.dt.day
df["week_day"] = df.date.dt.weekday
#%%

#split up in test and training data
df_train = df[df.Date < "2019"]
df_valid = df[df.Date >= "2019"]

exogenous_features = ["High_mean_lag3", "High_std_lag3", "Low_mean_lag3", "Low_std_lag3",
                      "Volume_mean_lag3", "Volume_std_lag3", "Turnover_mean_lag3",
                      "Turnover_std_lag3", "Trades_mean_lag3", "Trades_std_lag3",
                      "High_mean_lag7", "High_std_lag7", "Low_mean_lag7", "Low_std_lag7",
                      "Volume_mean_lag7", "Volume_std_lag7", "Turnover_mean_lag7",
                      "Turnover_std_lag7", "Trades_mean_lag7", "Trades_std_lag7",
                      "High_mean_lag30", "High_std_lag30", "Low_mean_lag30", "Low_std_lag30",
                      "Volume_mean_lag30", "Volume_std_lag30", "Turnover_mean_lag30",
                      "Turnover_std_lag30", "Trades_mean_lag30", "Trades_std_lag30",
                      "month", "week", "day", "day_of_week"]

#LOADING AND FITTING ARIMA -AUTO REGRESSIVE INTEGRATED MOVING AVERAGE- TO FORECAST PRICES
#arima uses its own preceding values lagging figures and lagging forecast errors to predict future values

model = auto_arima(df_train.VWAP, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.VWAP, exogenous=df_train[exogenous_features])

forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
df_valid["Forecast_ARIMAX"] = forecast
##RESULT: PICK THE MODEL WITH THE LOWEST AIC
#%%
#plot the results
df_valid[["VWAP", "Forecast_ARIMAX"]].plot(figsize=(14, 7))
#%%
print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_ARIMAX)))
print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_ARIMAX))
#%%
#load and fit facebook prophet
#prophet is an additive model for prediction of time series
#works best on time series that have seasonal effects and seasons of long historical series
