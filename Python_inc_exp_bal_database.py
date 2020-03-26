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
#from collections import defaultdict
#from datetime import datetime as dt
#import regex
import os
#%%
#determine the path of the files
path_win = os.path.relpath(r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx')
path_mac = os.path.relpath('/Users/bill/OneDrive - Envel/2020-01-28 envel.ai Working Class Sample.xlsx')
#read the original XLSX file and then split it up in 3 different dataframes
#no preprocessing here or encoding
df_card = pd.read_excel(path_win, sheet_name = "Card Panel")
df_bank = pd.read_excel(path_win, sheet_name = "Bank Panel")
#df_demo = pd.read_excel(path_win, sheet_name = "User Demographics")
#%%
#in the non-encoded verion all columns still have correct types
#extract unique numbers from all panels to find out unique users;
card_members = df_card['unique_mem_id'].unique()
bank_members = df_bank['unique_mem_id'].unique()
#demo_members = df_card['unique_mem_id'].unique()
trans_cat_card = df_card['transaction_category_name'].unique()
trans_cat_bank = df_bank['transaction_category_name'].unique()
#%%
'''
Following lists contains the categories to classify transactions either as expense or income
names taken directly from the Yodlee dataset; can be appended at will
'''
#append these unique to dictionaries measuring expenses or income with their respective categories
card_inc = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts']
card_exp = ['Groceries', 'Automotive/Fuel', 'Home Improvement', 'Travel',
            'Restaurants', 'Healthcare/Medical', 'Credit Card Payments',
            'Electronics/General Merchandise', 'Entertainment/Recreation',
            'Postage/Shipping', 'Other Expenses', 'Personal/Family',
            'Service Charges/Fees', 'Services/Supplies', 'Utilities',
            'Office Expenses', 'Cable/Satellite/Telecom',
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
'''
Iterate through rows and create new columns with a keyword that it is either an expense or income
This part is needed to make sure that initial balances can be determined better
'''
#DF_CARD
try:
    transaction_class_card = pd.Series([], dtype = 'object')
    for index, i in enumerate(df_card['transaction_category_name']):
        if i in card_inc:
            transaction_class_card[index] = "income"
        elif i in card_exp:
            transaction_class_card[index] = "expense"
        else:
            transaction_class_card[index] = "NOT_CLASSIFIED"
    df_card.insert(loc = len(df_card.columns), column = "transaction_class", value = transaction_class_card)
except:
    print("column is already existing or another error")
#    df_card.drop(['transaction_class'], axis = 1)
#    df_card.insert(loc = len(df_card.columns), column = "transaction_class", value = transaction_class_card)
###################################
#DF_BANK
try:
    transaction_class_bank = pd.Series([], dtype = 'object')
    for index, i in enumerate(df_bank['transaction_category_name']):
        if i in bank_inc:
            transaction_class_bank[index] = "income"
        elif i in bank_exp:
            transaction_class_bank[index] = "expense"
        else:
            transaction_class_bank[index] = "NOT_CLASSIFIED"
    df_bank.insert(loc = len(df_bank.columns), column = "transaction_class", value = transaction_class_bank)
except:
    print("column is already existing or another error")
#%%
'''
POSTGRE-SQL COLUMNS
This section adds a classification of transaction categories to allow a proper allocation to either the cash or the bills envelope
Bills describes as of 3/26/2020 all kinds of payment whose occurrence is beyond one's control,
that comes due and for which non-compliance has evere consequences
All other kinds of payments that are of optional nature and can be avoided are classifed as cash
'''
cash_env_card = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts',
                 'Restaurants', 'Electronics/General Merchandise',
                 'Entertainment/Recreation', 'Postage/Shipping', 'Other Expenses',
                 'Personal/Family','Groceries', 'Automotive/Fuel',  'Travel']

bill_env_card = ['Home Improvement', 'Healthcare/Medical', 'Credit Card Payments'
                 'Service Charges/Fees', 'Services/Supplies', 'Utilities',
                 'Office Expenses', 'Cable/Satellite/Telecom',
                 'Subscriptions/Renewals', 'Insurance']

cash_env_bank = ['Deposits', 'Salary/Regular Income', 'Transfers',
                 'Investment/Retirement Income', 'Rewards', 'Other Income',
                 'Refunds/Adjustments', 'Interest Income', 'Gifts', 'Expense Reimbursement',
                 'Electronics/General Merchandise', 'Groceries', 'Automotive/Fuel',
                 'Restaurants', 'Personal/Family', 'Entertainment/Recreation',
                 'Services/Supplies', 'Other Expenses', 'ATM/Cash Withdrawals',
                 'Postage/Shipping', 'Travel', 'Education', 'Charitable Giving',
                 'Office Expenses']

bill_env_bank = ['Service Charges/Fees', 'Credit Card Payments',
                 'Utilities', 'Healthcare/Medical', 'Loans', 'Check Payment',
                 'Cable/Satellite/Telecom', 'Insurance', 'Taxes', 'Home Improvement',
                 'Subscriptions/Renewals', 'Rent', 'Mortgage']
#iterate through rows and create a new columns with a note that it is either an expense or income
#DF_CARD
try:
    envelope_cat_card = pd.Series([], dtype = 'object')
    for index, i in enumerate(df_card['transaction_category_name']):
        if i in cash_env_card:
            envelope_cat_card[index] = "cash"
        elif i in bill_env_card:
            envelope_cat_card[index] = "bill"
        else:
            envelope_cat_card[index] = "NOT_CLASSIFIED"
    df_card.insert(loc = len(df_card.columns), column = "envelope_category", value = envelope_cat_card)
except:
    print("CASH/BILL column is already existing or another error")
##############################
#DF_BANK
try:
    envelope_cat_bank = pd.Series([], dtype = 'object')
    for i in enumerate(df_bank['transaction_category_name']):
        if i in cash_env_bank:
            envelope_cat_bank[index] = "cash"
        elif i in bill_env_bank:
            envelope_cat_bank[index] = "bill"
        else:
            envelope_cat_bank[index] = "NOT_CLASSIFIED"
    df_bank.insert(loc = len(df_bank.columns), column = "envelope_category", value = envelope_cat_bank)
except:
    print("CASH/BILL column is already existing or another error")
#%%
'''
Datetime engineering for card and bank panel
These columns help for reporting like weekly or monthly expenses and
improve prediction of re-occurring transactions
'''
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
'''
#Slices based on parameters
df[column][condition =<> x].sum()
totalMoney = df.Amount.sum()
totalGained = df["Amount"][df["Amount"] >= 0].sum()
'''
for mem_id in card_members:
    print(mem_id)
'''
70850441974905670928446
201492116860211330700059
257154737161372702866152
364987015290224198196263
651451454569880463282551
748150568877494117414131
'''
for mem_id in card_members:
    df_1 = df_card[['amount', 'envelope_category', 'transaction_class']][df_card['unique_mem_id'] == mem_id]
    print(df_mem_id)

#%%
#df = pd.read_csv("file")
#d= dict([(i,[a,b,c ]) for i, a,b,c in zip(df.ID, df.A,df.B,df.C)])
#%%
test_dictionary = {}
for ids, money in zip(df_card.unique_mem_id, df_card.amount):
    print(ids, money)
#%%
#test with csv
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
#get content of the entire row with the second element of the tuple that is being generated
#row = next(df_card.iterrows())[1]

#ALMOST WORKS
amount_list = []
for member in card_members:
    for index, row in df_card.iterrows():
         # access data using column names
         if row['transaction_class'] != "expense":
             #print(index, row['unique_mem_id'], row['amount'], row['transaction_class'])
             amount_list.append(row['amount'])
             cumulative_amount = np.cumsum(amount_list, axis = 0)
             print(f"INDEX:{index}, USER_ID:{row['unique_mem_id']}, \n {cumulative_amount}")
         #else:
          #   print("stopped at {row['index']}, user_ID: {row['unique_mem_id']}, cumulative sum injected: {cumulative amount}")
           #  break
#%%
#THIS MOFO WORKS
#for row in flights.head().itertuples():
#    print(row.Index, row.date, row.delay)
#in tuples slice columns with df.col instead of ['col']
amount_list = []

for member in card_members:
    for row in df_card.itertuples():
         # access data using column names
         if row.transaction_class == "expense":
             #print(index, row.unique_mem_id, row.amount, row.transaction_class)
             amount_list.append(row.amount)
             cumulative_amount = np.cumsum(amount_list, axis = 0)
             print(row.unique_mem_id, cumulative_amount)
         else:
             #print(row.unique_mem_id, cumulative_amount)
             print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
             break
    #print out the member id as part of the for-loop and and the last element of the list
    print(f"unique_member_ID: {member}; {cumulative_amount[-1]}")
#    cumulative_amount = []
#for member in card_members:
#    print(f"unique_member_ID: {member}; {cumulative_amount[-1]}")
#%%
'''
IMPROVISED SOLUTION WITHOUT ITERATION
Filter-df by unique id of each customer with columns: member_id; amount; envelope_category; transaction_class
iteration over each row as tuples and append amount to a list.
This list is taken and used for a cumulative sum of all transactions with type "expense"
Until "income" class is hit to stop
Numerical amount needs to be injected for simulation
'''
df_1 = df_card[['unique_mem_id', 'amount', 'envelope_category', 'transaction_class']][df_card['unique_mem_id'] == '70850441974905670928446']
df_2 = df_card[['unique_mem_id', 'amount', 'envelope_category', 'transaction_class']][df_card['unique_mem_id'] == '201492116860211330700059']
df_3 = df_card[['unique_mem_id', 'amount', 'envelope_category', 'transaction_class']][df_card['unique_mem_id'] == '257154737161372702866152']
df_4 = df_card[['unique_mem_id', 'amount', 'envelope_category', 'transaction_class']][df_card['unique_mem_id'] == '364987015290224198196263']
df_5 = df_card[['unique_mem_id', 'amount', 'envelope_category', 'transaction_class']][df_card['unique_mem_id'] == '651451454569880463282551']
df_6 = df_card[['unique_mem_id', 'amount', 'envelope_category', 'transaction_class']][df_card['unique_mem_id'] == '748150568877494117414131']
#DF_1
#for member in card_members:
cumulative_amount = []
amount_list = []
for row in df_1.itertuples():
    #access data using column names
    if row.transaction_class == "expense":
        #print(index, row.unique_mem_id, row.amount, row.transaction_class)
        amount_list.append(row.amount)
        cumulative_amount = np.cumsum(amount_list, axis = 0)
        print(row.unique_mem_id, cumulative_amount)
    else:
        print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
        break
    #print out the member id as part of the for-loop and and the last element of the list
print(f"unique_member_ID: {row.unique_mem_id}; {cumulative_amount[-1]}")
#%%
##DF_2
cumulative_amount = []
amount_list = []
for row in df_2.itertuples():
    #access data using column names
    if row.transaction_class == "expense":
        #print an overview and calculate the cumulative sum
        amount_list.append(row.amount)
        cumulative_amount = np.cumsum(amount_list, axis = 0)
        print(row.unique_mem_id, cumulative_amount)
    else:
        print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
        break
    #print out the member id as part of the for-loop and and the last element of the list
print(f"unique_member_ID: {row.unique_mem_id}; {cumulative_amount[-1]}")
#%%
##DF_3
cumulative_amount = []
amount_list = []
for row in df_3.itertuples():
    #access data using column names
    if row.transaction_class == "expense":
        #print an overview and calculate the cumulative sum
        amount_list.append(row.amount)
        cumulative_amount = np.cumsum(amount_list, axis = 0)
        print(row.unique_mem_id, cumulative_amount)
    else:
        print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
        break
    #print out the member id as part of the for-loop and and the last element of the list
print(f"unique_member_ID: {row.unique_mem_id}; {cumulative_amount[-1]}")
#%%
##DF_4
cumulative_amount = []
amount_list = []
for row in df_4.itertuples():
    #access data using column names
    if row.transaction_class == "expense":
        #print an overview and calculate the cumulative sum
        amount_list.append(row.amount)
        cumulative_amount = np.cumsum(amount_list, axis = 0)
        print(row.unique_mem_id, cumulative_amount)
    else:
        print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
        break
    #print out the member id as part of the for-loop and and the last element of the list
print(f"unique_member_ID: {row.unique_mem_id}; {cumulative_amount[-1]}")
#%%
##DF_5
cumulative_amount = []
amount_list = []
for row in df_5.itertuples():
    #access data using column names
    if row.transaction_class == "expense":
        #print an overview and calculate the cumulative sum
        amount_list.append(row.amount)
        cumulative_amount = np.cumsum(amount_list, axis = 0)
        print(row.unique_mem_id, cumulative_amount)
    else:
        print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
        break
    #print out the member id as part of the for-loop and and the last element of the list
print(f"unique_member_ID: {row.unique_mem_id}; {cumulative_amount[-1]}")
#%%
##DF_6
cumulative_amount = []
amount_list = []
for row in df_6.itertuples():
    #access data using column names
    if row.transaction_class == "expense":
        #print an overview and calculate the cumulative sum
        amount_list.append(row.amount)
        cumulative_amount = np.cumsum(amount_list, axis = 0)
        print(row.unique_mem_id, cumulative_amount)
    else:
        print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
        break
    #print out the member id as part of the for-loop and and the last element of the list
print(f"unique_member_ID: {row.unique_mem_id}; {cumulative_amount[-1]}")
#%%
#for row in flights.head().itertuples():
#    print(row.Index, row.date, row.delay)
amount_list = []
for member in card_members:
    for row in df_card.itertuples():
         # access data using column names
         if row.transaction_class != "expense":
             #print(index, row.unique_mem_id, row.amount, row.transaction_class)
             amount_list.append(row.amount)
             cumulative_amount = np.cumsum(amount_list, axis = 0)
             print(row.unique_mem_id, cumulative_amount)
         else:
             print(row.unique_mem_id, cumulative_amount)
             break
#%%
#almost works
#dictionary displays mem_id with transaction sum ( no split up between exp/inc)
#turnover_dictionary= {}
#for mem_id in card_members:
#    key = mem_id
#    value = df_card[df_card['unique_mem_id'] == mem_id]['amount'].sum()
#    turnover_dictionary[key] += value
#print(turnover_dictionary.keys())