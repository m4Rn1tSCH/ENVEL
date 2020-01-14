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
from fbprophet import Prophet
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
#df.set_index("Date", drop=False, inplace=True)
#df.head(n = 3)
'''
pick or rename columns accordingly
Index(['date', '1. open', '2. high', '3. low', '4. close', '5. volume'], dtype='object')
'''
#%%
#plotting the vwap
df.VWAP.plot(figsize = (14, 7))
#%%
#FEATURE ENGINEERING
#this is done to enhance the models prediction ability and accuracy
#typical engineered features based on lagging metrics
#mean + stdev of past 3d/7d/30d/ + rolling volume

df.reset_index(drop = True, inplace = True)
lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
time1 = 3
time2 = 7
time3 = 30

#rolling volume of HLVTT
df_rolled_3d = df[lag_features].rolling(window = time1, min_periods = 0)
df_rolled_7d = df[lag_features].rolling(window = time2, min_periods = 0)
df_rolled_30d = df[lag_features].rolling(window = time3, min_periods = 0)

#
df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df[f"{feature}_mean_lag{time1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{time2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{time3}"] = df_mean_30d[feature]

    df[f"{feature}_std_lag{time1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{time2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{time3}"] = df_std_30d[feature]

df.fillna(df.mean(), inplace=True)

df.set_index("Date", drop=False, inplace=True)
df.head(n = 3)
#%%
#add daytime features to add additional info the model can be fed
df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
df["month"] = df.Date.dt.month
df["week"] = df.Date.dt.week
df["day"] = df.Date.dt.day
df["day_of_week"] = df.Date.dt.dayofweek
df.head()

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
