# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:01:06 2020

@author: bill-
"""

'''
Purpose of this script is to interact with a csv file/database and produce a dictionary
with unique IDs and corresponding income and expenses in separate dictionaries
-determine income and expenses based on given categories
-add it either to the INCOME DICTIONARY or the EXPENSE DICTIONARY
-find out daily, weekly and monthly throughput of accounts and their excess cash
-develop a logic or daily limits and spending patterns
'''
#load needed packages
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime as dt
import regex
import os
#%%
#determine the path of the files
path_win = os.path.relpath(r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx')
path_mac = os.path.relpath('/Users/bill/OneDrive - Envel/2020-01-28 envel.ai Working Class Sample.xlsx')
#read the original XLSX file and then split it up in 3 different dataframes
#no preprocessing here or encoding
df_card = pd.read_excel(path_mac, sheet_name = "Card Panel")
df_bank = pd.read_excel(path_mac, sheet_name = "Bank Panel")
df_demo = pd.read_excel(path_mac, sheet_name = "User Demographics")
#%%
#in the non-encoded verion all columns still have correct types
#extract unique numbers from all panels to find out unique users;
card_members = df_card['unique_mem_id'].unique()
bank_members = df_bank['unique_mem_id'].unique()
demo_members = df_card['unique_mem_id'].unique()
trans_cat_card = df_card['transaction_category_name'].unique()
trans_cat_bank = df_bank['transaction_category_name'].unique()
#append these unique to dictionaries measuring expenses or income with their respective categories
card_inc = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts']
card_exp = ['Groceries' 'Automotive/Fuel' 'Home Improvement' 'Travel' 'Restaurants'
 'Healthcare/Medical' 'Credit Card Payments'
 'Electronics/General Merchandise' 'Entertainment/Recreation'
 'Postage/Shipping' 'Other Expenses' 'Personal/Family'
 'Service Charges/Fees'  'Services/Supplies', 'Utilities'
 'Office Expenses' 'Cable/Satellite/Telecom',
 'Subscriptions/Renewals', 'Insurance']
bank_inc = ['Deposits', 'Salary/Regular Income', 'Transfers',
            'Investment/Retirement Income', 'Rewards', 'Other Income',
            'Refunds/Adjustments', 'Interest Income', 'Gifts', 'Expense Reimbursement']
bank_exp = ['Service Charges/Fees',
            'Credit Card Payments', 'Utilities', 'Healthcare/Medical', 'Loans',
            'Check Payment', 'Electronics/General Merchandise', 'Groceries',
            'Automotive/Fuel', 'Restaurants', 'Personal/Family',
            'Entertainment/Recreation', 'Services/Supplies', 'Other Expenses',
            'ATM/Cash Withdrawals', 'Cable/Satellite/Telecom',
            'Postage/Shipping', 'Insurance', 'Travel', 'Taxes',
            'Home Improvement', 'Education', 'Charitable Giving',
            'Subscriptions/Renewals', 'Rent', 'Office Expenses', 'Mortgage']
#%%
#iterate through rows and create a new columns with a note that it is either an expense or income
#DF_CARD
try:
    transaction_class_card = pd.Series([], dtype = 'object')
    for i in range(len(df_card)):
        if df_card["transaction_category_name"][i] in card_inc:
            transaction_class_card[i] = "income"
        elif df_card["transaction_category_name"][i] in card_exp:
            transaction_class_card[i] = "expense"
        else:
            transaction_class_card[i] = "NOT CLASSIFIED"
    df_card.insert(loc = len(df_card.columns), column = "transaction_class", value = transaction_class_card)
except:
    print("column is already existing, canot be added again")
#    df_card.drop(['transaction_class'], axis = 1)
#    df_card.insert(loc = len(df_card.columns), column = "transaction_class", value = transaction_class_card)
#%%
#DF_BANK
try:
    transaction_class_bank = pd.Series([], dtype = 'object')
    for i in range(len(df_bank)):
        if df_bank["transaction_category_name"][i] in bank_inc:
            transaction_class_bank[i] = "income"
        elif df_bank["transaction_category_name"][i] in bank_exp:
            transaction_class_bank[i] = "expense"
        else:
            transaction_class_bank[i] = "NOT CLASSIFIED"
    df_bank.insert(loc = len(df_bank.columns), column = "transaction_class", value = transaction_class_bank)
except:
    print("column is already existing and cannot be appended again")
#    df_card.drop(['transaction_class'], axis = 1)
#    df_card.insert(loc = len(df_card.columns), column = "transaction_class", value = transaction_class_card)
#%%
#Datetime engineering DF_CARD
for col in list(df_card):
    if df_card[col].dtype == 'datetime64[ns]':
        df_card[f"{col}_month"] = df_card[col].dt.month
        df_card[f"{col}_week"] = df_card[col].dt.week
        df_card[f"{col}_weekday"] = df_card[col].dt.weekday
#%%
#Datetime engineering DF_BANK
for col in list(df_bank):
    if df_bank[col].dtype == 'datetime64[ns]':
        df_bank[f"{col}_month"] = df_bank[col].dt.month
        df_bank[f"{col}_week"] = df_bank[col].dt.week
        df_bank[f"{col}_weekday"] = df_bank[col].dt.weekday
#%%
#DATETIME ENGINEERING
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
#DataFrame.shift(self, periods = 1, freq = None, axis = 0, fill_value = None)
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
#DATETIME ENGINEERING
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
#Add feature columns for additive spending on a weekly; monthly; daily basis
#total throughput of money
total_throughput = df_card['amount'].sum()
#monthly figures
net_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).sum()
avg_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).mean()
#CHECK VIABILITY OF SUCH VARIABLES
monthly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
#monthly_expenses = df_card['amount'][df_card['transaction_base_type'] == 'debit'].groupby(df_card['transaction_date_week']).sum()
#weekly figures
net_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).sum()
avg_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).mean()
#CHECK VIABILITY OF SUCH VARIABLES
weekly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
#weekly_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_week']).sum()
#daily figures
net_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).mean()
avg_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).sum()
#CHECK VIABILITY OF SUCH VARIABLES
daily_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_weekday']).sum()
#daily_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_weekday']).sum()
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
#%%
#append to lists whether it is income or expense
#filter with ilocation and show expenses and income as spearate dataframe
card_expenses = df_card.iloc[np.where(df_card['transaction_class'] == "expense")]
card_income = df_card.iloc[np.where(df_card['transaction_class'] == "income")]
bank_expenses = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")]
bank_income = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")]
#%%
#Create an empty dictionary: income and expenses
income_dict = {}

#Loop over dataframe for card transactions
#tuple indices have to be numbers!!
for mem_id, amount in df_card.items():
    #Add the unique member ID as key and one or several columns as values
    #Dict[Key] = Value
    income_dict[mem_id] =  amount, #amount_mean_lag30
print(income_dict.keys())

#Sort the names list by rank in descending order and slice the first 10 items
#for row in sorted(row, reverse = False)[:10]:
#    #Print each item
#    print(income_dict[card_members])
#%%
#to_dict method
#df_income = df_card[['unique_mem_id', 'amount']]
#test_dict = df_income.T.to_dict()
#%%
#df = pd.read_csv("file")
#d= dict([(i,[a,b,c ]) for i, a,b,c in zip(df.ID, df.A,df.B,df.C)])
#test_dictionary = {}
#for i, income in zip(df_card.unique_mem_id, df_card.amount):
#    print(set(zip(df_card.unique_mem_id, df_card.amount))
#    test_dictionary[i] = income
#%%
#try with csv here
#from collections import defaultdict

#d = defaultdict(int)

#with open("data.txt") as f:
#    for line in f:
#        tokens = [t.strip() for t in line.split(",")]
#        try:
#            sid = int(tokens[3])
#            mark = int(tokens[4])
#        except ValueError:
#            continue
#        d[sid] += mark
#print(d)
#%%
#try directly here
d = defaultdict(list)

for row in df_card:
    try:
        key = card_members
        value = df_card['amount']
    except ValueError:
        value = str('not classified')
        continue
    d[key] += value
print(d)
#%%
#test_dictionary = {}

for ids, money in zip(df_card.unique_mem_id, df_card.amount):
    print(ids, money)
#%%
#df.set_index('ID').T.to_dict('list')
#{s'p': [1, 3, 2], 'q': [4, 3, 2], 'r': [4, 0, 9]}
#%%
#tst with csv
#from collections import defaultdict

#d = defaultdict(int)

#with open("data.txt") as f:
#    for line in f:
#        tokens = [t.strip() for t in line.split(",")]
#        try:
#            key = int(tokens[3])
#            value = int(tokens[4])
#        except ValueError:
#            continue
#        d[key] += value
#print d
#%%
#test directly
from collections import defaultdict
trans_dict = defaultdict(list)

for row in df_card.items():
    try:
        key = row[0]
        value = row[3]
    except ValueError:
        continue
    trans_dict[key] += value

#print(trans_dict)
#%%
filter(lambda line: line != '!', open('something.txt'))
#%%
for mem_id in card_members:
    print(mem_id)
    print(df_card[df_card['unique_mem_id'] == mem_id]['amount'].sum())
#%%
#dictionary displays mem_id with transaction sum ( no split up between exp/inc)
turnover_dictionary= {}
for mem_id in card_members:
    key = mem_id
    value = df_card[df_card['unique_mem_id'] == mem_id]['amount'].sum()
    turnover_dictionary[key] = value
print(turnover_dictionary.keys())
#%%
dictionary = {}
for mem_id in card_members:
    key = mem_id
    value = df_card[df_card['unique_mem_id'] == mem_id]['amount'].sum()
    dictionary[key] = value
print(dictionary.keys())
#%%
dictionary = {}
for mem_id in card_members:
    for row in df_card[df_card['transaction_class'] == 'expense']:
        key = mem_id
        value = df_card[df_card['unique_mem_id'] == mem_id]['amount']
        dictionary[key] += value
    else:
        break
print(dictionary.keys())
#%%
mem_id_df = df_card[['unique_mem_id', 'amount']].groupby('unique_mem_id').apply(lambda x: x['unique_mem_id'].unique())
#%%
mem_id_df_2 = df_card[['unique_mem_id', 'amount']].groupby('unique_mem_id')
print(pd.DataFrame(mem_id_df_2))
#%%
mem_id_d = dict(zip(df_card.unique_mem_id, df_card.amount))
#%%
transaction_dict = {}
expenses_only = df_card[df_card['transaction_class'] == 'expense']
zip(expenses_only)
for unique_mem_id, amount in expenses_only.items():
    transaction_dict[unique_mem_id[0]] = amount[3]
#%%
while df_card[df_card['transaction_class'] != 'income']:
    
