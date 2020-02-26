# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:49:25 2020

@author: bill-
"""
#load packages
import pandas as pd
import os
import matplotlib.pyplot as plt

#from datetime import datetime
#import seaborn as sns
#plt.rcParams["figure.dpi"] = 600
#plt.rcParams['figure.figsize'] = [12, 10]
#%%
'''
Setup of the function to merge every single operation into one function that is then called by the flask connection/or SQL
contains: preprocessing, splitting, training and eventually prediction
'''
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
#$ export FLASK_APP = C:\Users\bill-\OneDrive\Dokumente\Docs Bill\TA_files\functions_scripts_storage\Python_prediction_merged_function_database.py
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
preprocessed_input = ''.join(('', path_1, ''))
'''
SCRIPT WILL GET ALL XLSX SHEETS AT THIS STAGE!
'''
#relative_t_path = './*.csv'
#%%
def predict_needed_value(preprocessed_input):
    #%%
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
#Add feature columns for additive spending on a weekly; monthly; daily basis
#total throughput of money
total_throughput = df_card['amount'].sum()
#monthly figures
net_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).sum()
avg_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).mean()
monthly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
monthly_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_week']).sum()
#weekly figures
net_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).sum()
avg_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).mean()
weekly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
weekly_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_week']).sum()
#daily figures
net_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).mean()
avg_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).sum()
daily_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_weekday']).sum()
daily_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_weekday']).sum()
#%%
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
