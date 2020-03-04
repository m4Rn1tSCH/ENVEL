# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:46:28 2020

@author: bill-
"""
#load packages
import pandas as pd
import numpy as np
import os
import random as rd
import string

import matplotlib.pyplot as plt
from datetime import datetime as dt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import seaborn as sns
plt.rcParams["figure.dpi"] = 600
plt.rcParams['figure.figsize'] = [12, 10]
#%%
#CONNECTION TO FLASK/SQL
#INSERT FLASK CONNECTION SCRIPT HERE
###########################################
#loading the simplified applications
#from flask import Flask
#app = Flask(__Preprocessor__)

##put address here
#@app.route('/')
#def hello_world():
#    return 'Hello, World!'
#route tells what URL should trigger the function
#use __main__ only for the actual core of the application
# pick unique names for particular functions if these are imported later
#DONT CALL APPLICATIONS FLASK.PY TO AVOID CONFLICTS WITH FLASK

#RUN THE APPLICATION
#flask command or -m swith in Python

########SETTING THE ENVIRONMENT VARIABLE#######
#$ export FLASK_APP = C:\Users\bill-\OneDrive\Dokumente\Docs Bill\TA_files\functions_scripts_storage\Python_preprocessing_merged_function_database.py
#$ flask run
# * Running on http://127.0.0.1:5000/

####COMMAND PROMPT#####
#C:\path\to\app>set FLASK_APP=hello.py

####for production use##
#$ flask run --host=0.0.0.0
############################################
#INSERT SQL CONNECTION HERE
############################################
###SQL-CONNECTION TO QUERY THE VENDOR FILE
###Create engine
##engine = create_engine('sqlite:///Chinook.sqlite')

##Open engine connection: con
##con = engine.connect()

##Perform query: rs
##rs = con.execute("SELECT * from <<DB_FOLDER>>")

#Save results df
##df = pd.DataFrame(rs.fetchall())

##Close connection
##con.close()
##############################################
#%%
######LOADING THE TRANSACTION FILE#####
transaction_file = r"C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx"
path_1 = transaction_file.replace(os.sep,'/')
transactions = ''.join(('', path_1, ''))
'''
SCRIPT WILL GET ALL XLSX SHEETS AT THIS STAGE!
'''
#relative_t_path = './*.csv'
#%%
def random_id(length):
    letters = string.ascii_letters
    numbers = string.digits
    result_str = ''.join(rd.choice(letters) for n in range(length))
    result_num = ''.join(rd.choice(numbers) for n in range(length))
    unique_id = ''.join(result_str) + ''.join(result_num)
    return unique_id
#%%
def preprocess_input_file(transactions):
    df_card = pd.read_excel(transactions, sheet_name = "Card Panel")
    df_bank = pd.read_excel(transactions, sheet_name = "Bank Panel")
    df_demo = pd.read_excel(transactions, sheet_name = "User Demographics")
    #info
    df_card.info()
    df_card.describe()
    #takes 10 minutes!!
    #sns.pairplot(df_card)
    print(df_card.head(3))
    print("--------------------------------------------")
    df_bank.info()
    df_bank.describe()
    #takes 10 minutes!!
    #sns.pairplot(df_bank)
    print(df_bank.head(3))
    print("--------------------------------------------")
    df_demo.info()
    print(df_demo.head(3))
    print("--------------------------------------------")
#%%
    #FEATURE ENGINEERING
    #Add date feature columns to the card_panel sheet
    #this is done to enhance the models prediction ability and accuracy
    for col in list(df_card):
        if df_card[col].dtype == 'datetime64[ns]':
            df_card[f"{col}_month"] = df_card[col].dt.month
            df_card[f"{col}_week"] = df_card[col].dt.week
            df_card[f"{col}_weekday"] = df_card[col].dt.weekday
#%%
    #FEATURE ENGINEERING
    #Add date feature columns to the bank_panel sheet
    #this is done to enhance the models prediction ability and accuracy
    for col in list(df_bank):
        if df_bank[col].dtype == 'datetime64[ns]':
            df_bank[f"{col}_month"] = df_bank[col].dt.month
            df_bank[f"{col}_week"] = df_bank[col].dt.week
            df_bank[f"{col}_weekday"] = df_bank[col].dt.weekday
#%%
    #FEATURE ENGINEERING II
    #typical engineered features based on lagging metrics
    #mean + stdev of past 3d/7d/30d/ + rolling volume
    df_card.reset_index(drop = True, inplace = True)
    #pick lag features to iterate through and calculate features
    #original lag features; based on tutorial dataset
    #lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
    lag_features = ["amount"]
    #set up time frames; how many days/months back/forth
    t1 = 3
    t2 = 7
    t3 = 30
    #rolling values for all columns ready to be processed
    #DataFrame.rolling(self, window, min_periods = None, center = False, win_type = None, on = None, axis = 0, closed = None)
    #rolling method; window = size of the moving window;
                    #min_periods = min no. of obersvations that need to have a value(otherwise result is NA)
                    #center = set labels at the center of the window
                    #win_type = weighting of points, "None" all points are equally weighted
                    #on = use datetime-like column index (instead of df indices) to calculate the value
                    #axis = 0:row-wise; 1:column-wise
                    #closed = ['right', 'left', 'both', 'neither'] close of the interval; for offset-based windows defaults to rights;
                    #for fixed windows defaults to both
    #DataFrame.shift(self, periods = 1, freq = None, axis = 0, fill_value = None)
                    #periods = pos/ neg downwards or upwards shift in periods
                    #freq = offset/timedelta/str; index shifted but data not realigned; extend index when shifting + preserve original data
                    #axis = shift direction (0: index 1: columns None)
                    #fill_value = numeric: np.nan; datetime,timedelta: NaT; extension types:dtype.na_value
    df_card_rolled_3d = df_card[lag_features].rolling(window = t1, min_periods = 0)
    df_card_rolled_7d = df_card[lag_features].rolling(window = t2, min_periods = 0)
    df_card_rolled_30d = df_card[lag_features].rolling(window = t3, min_periods = 0)

    #calculate the mean with a shifting time window
    df_card_mean_3d = df_card_rolled_3d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_card_mean_7d = df_card_rolled_7d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_card_mean_30d = df_card_rolled_30d.mean().shift(periods = 1).reset_index().astype(np.float32)

    #calculate the std dev with a shifting time window
    df_card_std_3d = df_card_rolled_3d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_card_std_7d = df_card_rolled_7d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_card_std_30d = df_card_rolled_30d.std().shift(periods = 1).reset_index().astype(np.float32)

    for feature in lag_features:
        df_card[f"{feature}_mean_lag{t1}"] = df_card_mean_3d[feature]
        df_card[f"{feature}_mean_lag{t2}"] = df_card_mean_7d[feature]
        df_card[f"{feature}_mean_lag{t3}"] = df_card_mean_30d[feature]

        df_card[f"{feature}_std_lag{t1}"] = df_card_std_3d[feature]
        df_card[f"{feature}_std_lag{t2}"] = df_card_std_7d[feature]
        df_card[f"{feature}_std_lag{t3}"] = df_card_std_30d[feature]

    #fill missing values with the mean to keep distortion very low and allow prediction
    df_card.fillna(df_card.mean(), inplace = True)
    #associate date as the index columns to columns (especially the newly generated ones to allow navigating and slicing)
    df_card.set_index("transaction_date", drop = False, inplace = True)
#%%
        #FEATURE ENGINEERING II
    #typical engineered features based on lagging metrics
    #mean + stdev of past 3d/7d/30d/ + rolling volume
    df_bank.reset_index(drop = True, inplace = True)
    #pick lag features to iterate through and calculate features
    #original lag features; based on tutorial dataset
    #lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
    lag_features = ["amount"]
    #set up time frames; how many days/months back/forth
    t1 = 3
    t2 = 7
    t3 = 30
    #rolling values for all columns ready to be processed
    #DataFrame.rolling(self, window, min_periods = None, center = False, win_type = None, on = None, axis = 0, closed = None)
    #rolling method; window = size of the moving window;
                    #min_periods = min no. of obersvations that need to have a value(otherwise result is NA)
                    #center = set labels at the center of the window
                    #win_type = weighting of points, "None" all points are equally weighted
                    #on = use datetime-like column index (instead of df indices) to calculate the value
                    #axis = 0:row-wise; 1:column-wise
                    #closed = ['right', 'left', 'both', 'neither'] close of the interval; for offset-based windows defaults to rights;
                    #for fixed windows defaults to both
    #DataFrame.shift(self, periods = 1, freq = None, axis = 0, fill_value = None)
                    #periods = pos/ neg downwards or upwards shift in periods
                    #freq = offset/timedelta/str; index shifted but data not realigned; extend index when shifting + preserve original data
                    #axis = shift direction (0: index 1: columns None)
                    #fill_value = numeric: np.nan; datetime,timedelta: NaT; extension types:dtype.na_value
    df_bank_rolled_3d = df_bank[lag_features].rolling(window = t1, min_periods = 0)
    df_bank_rolled_7d = df_bank[lag_features].rolling(window = t2, min_periods = 0)
    df_bank_rolled_30d = df_bank[lag_features].rolling(window = t3, min_periods = 0)

    #calculate the mean with a shifting time window
    df_bank_mean_3d = df_bank_rolled_3d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_bank_mean_7d = df_bank_rolled_7d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_bank_mean_30d = df_bank_rolled_30d.mean().shift(periods = 1).reset_index().astype(np.float32)

    #calculate the std dev with a shifting time window
    df_bank_std_3d = df_bank_rolled_3d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_bank_std_7d = df_bank_rolled_7d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_bank_std_30d = df_bank_rolled_30d.std().shift(periods = 1).reset_index().astype(np.float32)

    for feature in lag_features:
        df_bank[f"{feature}_mean_lag{t1}"] = df_bank_mean_3d[feature]
        df_bank[f"{feature}_mean_lag{t2}"] = df_bank_mean_7d[feature]
        df_bank[f"{feature}_mean_lag{t3}"] = df_bank_mean_30d[feature]

        df_bank[f"{feature}_std_lag{t1}"] = df_bank_std_3d[feature]
        df_bank[f"{feature}_std_lag{t2}"] = df_bank_std_7d[feature]
        df_bank[f"{feature}_std_lag{t3}"] = df_bank_std_30d[feature]

    #fill missing values with the mean to keep distortion very low and allow prediction
    df_bank.fillna(df_bank.mean(), inplace = True)
    #associate date as the index columns to columns (especially the newly generated ones to allow navigating and slicing)
    df_bank.set_index("transaction_date", drop = False, inplace = True)
#%%
    #HOLIDAY CHECK FOR FEATURE ENGINEERING
    '''
    The idea is to have it as an engineered feature to find anomalies which might be attributable to holidays or pre-holiday periods
    and improve prediction results
    Next to the US calendar, users can import a calendar of their choice to have access to holidays from various regions
    limit adaption possible for holidays; reminders for shopping holidays about budget; specialized messages/notifications about holidays
    '''
    #today's date as string
    today = dt.today().strftime('%m/%d/%Y')

    #format is MM/DD/YYYY
    #given in string with leading zero for single-digit month
    holidays = ['04/01/2020', '05/29/2020', '12/31/2020', '02/21/2020', '02/25/2020']
    #separate test container for a picked local calendar for example
    test = []
    #iterate over and append to test list
    for x in holidays:
        test.append(x)
    if today in test:
        print(f"Today is {today}; have fun and enjoy your holiday :)")
#%%
    #REPORTING & ANALYSIS FEATURE FOR USERS
    #Add feature columns for additive spending on a weekly; monthly; daily basis
    #total throughput of money
    total_throughput = df_card['amount'].sum()
    #monthly figures
    net_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).sum()
    avg_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    monthly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
    monthly_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_week']).sum()
    #weekly figures
    net_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).sum()
    avg_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    weekly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
    weekly_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_week']).sum()
    #daily figures
    net_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).mean()
    avg_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).sum()
    #CHECK VIABILITY OF SUCH VARIABLES
    daily_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_weekday']).sum()
    daily_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_weekday']).sum()
    #report for users about their spending patterns, given in various intervals
    try:
        print(f"The total turnover on your account has been ${total_throughput}")
        print("................................................................")
        spending_metrics_monthly = pd.DataFrame(data = {'Average Monthly Spending':avg_monthly_throughput,
                                                        'Monthly Turnover':net_monthly_throughput})
        print(spending_metrics_monthly)
        print(".................................................................")
        spending_metrics_weekly = pd.DataFrame(data = {'Average Weekly Spending':avg_weekly_throughput,
                                                       'Weekly Turnover':net_weekly_throughput})
        print(spending_metrics_weekly)
        print(".................................................................")
        spending_metrics_daily = pd.DataFrame(data = {'Average Daily Spending':avg_daily_spending,
                                                      'Daily Turnover':net_daily_spending})
        print(spending_metrics_daily)
    except:
        print("You do not have enough transactions yet. But we are getting there...")
    #PLOTTING OF THE ORIGINAL/ENGINEERED FEATURES
    #the figure has to be created in the same cell/section as the axes with values!!
    #fig, ax = plt.subplots(2, 2, figsize = (20, 12))
    #pick the graph from top to bottom
    #DONT PICK COORDINATES LIKE ax[row_pos][col_pos] when column arg not 2
    #picking starts from top left to bottom right
    #ax[0][0].plot(df_card.index.values, df_card['amount'])
    #ax[0][1].plot(df_card.index.values, df_card['account_score'])
    #ax[1][0].plot(df_bank.index.values, df_bank['amount'])
    #ax[1][1].plot(df_bank.index.values, df_bank['account_score'])
    #SELECTION OF FEATURES AND LABELS TO PREDICT
    #V1
    #plan for features + prediction
    #conversion of df_card; df_bank; df_demo
#%%
    #CHECK FOR MISSING VALUES
    '''
    find missing values and mark the corresponding column as target that is to be predicted
    '''
    #iterate over all columns and search for missing values
    #find such missing values and declare it the target value
    #df in use is pandas datafame, use .iloc[]
    #df is a dictionary, .get()
    #iterate over columns first to find missing targets
    #iterate over rows of the specific column that has missing values
    #fill the missing values with a value
    y = []
    X = []
    for col in list(df_card):
        if df_card[col].isnull().any() == True:
            print(f"{col} is target variable and will be used for prediction")
            y.append(df_card[col])
            for index, row in df_card.iterrows():
                if row.isnull().any() == True:
                    print(f"Value missing in row {index}")
                    #df_card.loc[row].drop_duplicates(method = bfill)
                else:
                    print("Data set contains no missing values; specify the label manually")
                    pass
#%%
    #V2
    #first prediction loop and stop
    y = []
    X = []
    for col in list(df_card):
        if df_card[col].isnull().any() == True:
            print(f"{col} is target variable and will be used for prediction")
            y.append(df_card[col])
            if len(y) == 1:
                print("first prediction target found...")
                break
#%%
    #LABEL ENCODER for column card panel
    '''
    encode all non-numerical values to ready up the data set for classification and regression purposes
    '''
    #applying fit_transform yields: encoding of 22 columns but most of them remain int32 or int64
    #applying first fit to train the data and then apply transform will encode only 11 columns and leaves the others unchanged
    #if 2 or fewer unique categories data type changes to "object"
    #iterate through columns and change the object (not int64)

    ###ATTENTION; WHEN THE DATA FRAME IS OPEN AND THE SCRIPT IS RUN
    ###THE DATA TYPES CHANGE TO FLOAT64 AS THE NUMBERS ARE BEING DISPLAYED
    le = LabelEncoder()
    le_count = 0
    #Iterate through the columns
    #Train on the training data
    #Transform both training and testing
    #Keep track of how many columns were converted
    #fit first (dont create a column yet)
    #transform and overwrite column or create a new one
    try:
        for col in list(df_card):
            if df_card[col].dtype == 'object':
                le.fit(df_card[col])
                df_card[col] = le.transform(df_card[col])
                le_count += 1
    except:
        print(f"({df_card[col]} could not be converted")
    print('%d columns were converted.' % le_count)
    print("--------------------------------------------")
    #for comparison of the old data frame and the new one
    print("PROCESSED DATAFRAME:")
    print(df_card.head(3))
    print("DF card panel preprocessing finished + Ready for a report + Pass to ML models")
#%%
    #LABEL ENCODER for column bank panel
    '''
    encode all non-numerical values to ready up the data set for classification and regression purposes
    '''
    #applying fit_transform yields: encoding of 22 columns but most of them remain int32 or int64
    #applying first fit to train the data and then apply transform will encode only 11 columns and leaves the others unchanged
    #if 2 or fewer unique categories data type changes to "object"
    #iterate through columns and change the object (not int64)

    ###ATTENTION; WHEN THE DATA FRAME IS OPEN AND THE SCRIPT IS RUN
    ###THE DATA TYPES CHANGE TO FLOAT64 AS THE NUMBERS ARE BEING DISPLAYED
    le = LabelEncoder()
    le_count = 0
    #Iterate through the columns
    #Train on the training data
    #Transform both training and testing
    #Keep track of how many columns were converted
    #fit first (dont create a column yet)
    #transform and overwrite column or create a new one
    try:
        for col in list(df_bank):
            if df_bank[col].dtype == 'object':
                le.fit(df_bank[col])
                df_bank[col] = le.transform(df_bank[col])
                le_count += 1
    except:
        print(f"({df_bank[col]} could not be converted")
    print('%d columns were converted.' % le_count)
    print("--------------------------------------------")
    #for comparison of the old data frame and the new one
    print("PROCESSED DATAFRAME:")
    print(df_bank.head(3))
    print("DF bank panel preprocessing finished + Ready for a report + Pass to ML models")
#%%
##CONVERSION TO CSV FOR TESTING
'''
For testing purposes which does not include randomized IDs as part of the name and allows loading a constant name
Should this be generated as csv and sit in AWS for a daily report?
Should it be uploaded to the SQL database again?
'''
#The dataframe is now preprocessed and ready to be loaded by the prediction models for predictive analysis
#Conversion of df to CSV or direct pass possible
##Conversion
raw = 'C:/Users/bill-/Desktop/'
#raw_2 = 'C:/Users/bill-/OneDrive - Education First/Documents/Docs Bill/FILES_ENVEL/'
#working path = 'C:/Users/bill-/OneDrive/Dokumente/Docs Bill/TA_files/functions_scripts_storage/yodlee_converted.csv'
date_of_creation = dt.today().strftime('%m-%d-%Y')
#os.mkdir(os.path.join(raw, converted_csv))
csv_path_card = os.path.join(raw, date_of_creation + '_CARD_PANEL' + 'ID_' + '.csv')
csv_path_bank = os.path.join(raw, date_of_creation + '_BANK_PANEL' + 'ID_' + '.csv')
csv_path_demo = os.path.join(raw, date_of_creation + '_DEMO_PANEL' + 'ID_' + '.csv')
#name_list = ['df_card', 'df_bank', 'df_demo']
#for name in name_list:
#    f"{date_of_creation}_{name}".to_csv(csv_path)
df_card.to_csv(csv_path_card)
df_bank.to_csv(csv_path_bank)
df_demo.to_csv(csv_path_demo)
#%%
##CONVERSION TO CSV FINAL
#The dataframe is now preprocessed and ready to be loaded by the prediction models for predictive analysis
#Conversion of df to CSV or direct pass possible

#raw = 'C:/Users/bill-/Desktop/'
#working directory = 'C:/Users/bill-/OneDrive/Dokumente/Docs Bill/TA_files/functions_scripts_storage/yodlee_converted.csv'
#date_of_creation = dt.today().strftime('%m-%d-%Y')
#setting up paths
#csv_path_card = os.path.join(raw, date_of_creation + '_CARD_PANEL' + 'ID_' + random_id(length = 5) + '.csv')
#csv_path_bank = os.path.join(raw, date_of_creation + '_BANK_PANEL' + 'ID_' + random_id(length = 5) + '.csv')
#csv_path_demo = os.path.join(raw, date_of_creation + '_DEMO_PANEL' + 'ID_' + random_id(length = 5) + '.csv')
#converting
#df_card.to_csv(csv_path_card)
#df_bank.to_csv(csv_path_bank)
#df_demo.to_csv(csv_path_demo)
