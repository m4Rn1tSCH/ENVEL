# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:51:04 2020

@author: bill-
"""
'''
EDA module for various Yodlee datafeames
'''

#load needed packages
import pandas as pd
import numpy as np
from datetime import datetime as dt
from flask import Flask
import os
import csv
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest , chi2, f_classif
from sklearn.preprocessing import LabelEncoder

#imported custom function
#generates a CSV for daily/weekly/monthly account throughput; expenses and income
from Python_spending_report_csv_export_function import spending_report
#contains the connection script
from Python_SQL_connection import execute_read_query, create_connection, close_connection
#contains all credentials
import PostgreSQL_credentials as acc
#%%
#Py_SQL_con needs to be loaded first
connection = create_connection(db_name = acc.YDB_name, db_user = acc.YDB_user, db_password = acc.YDB_password, db_host = acc.YDB_host, db_port = acc.YDB_port)
#%%
#estbalish connection to get user IDs
filter_query = f"SELECT unique_mem_id, state, city, zip_code, income_class, file_created_date FROM user_demographic WHERE state = 'MA'"
transaction_query = execute_read_query(connection, filter_query)
query_df = pd.DataFrame(transaction_query,
                        columns = ['unique_mem_id', 'state', 'city', 'zip_code', 'income_class', 'file_created_date'])
#%%
#dateframe to gather MA bank data
try:
    for i in query_df['unique_mem_id'].sample(n = 1, random_state = 2):
        print(i)
        filter_query = f"SELECT * FROM bank_record WHERE unique_mem_id = '{i}'"
        transaction_query = execute_read_query(connection, filter_query)
        bank_df = pd.DataFrame(transaction_query,
                        columns = ['unique_mem_id', 'unique_bank_account_id', 'unique_bank_account_id','amount',
                                   'currency', 'description', 'transaction_date', 'post_date',
                                   'transaction_base_type', 'transaction_category_name', 'primary_merchant_name',
                                   'secondary_merchant_name', 'city','state', 'zip_code', 'transaction_origin',
                                   'factual_category', 'factual_id', 'file_created_date', 'optimized_transaction_date',
                                   'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred', 'swipe_date',
                                   'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
                                   'account_type', 'account_source_type', 'account_score', 'user_score', 'lag', 'is_duplicate'])
        print(f"User {i} has {len(bank_df)} transactions on record.")
except OperationalError as e:
        print(f"The error '{e}' occurred")
        connection.rollback
#%%
#Plot template
fig, ax = plt.subplots(2, 1, figsize = (25, 25))
ax[0].plot(df.index.values, df['x'], color = 'green', lw = 4, ls = '-.', marker = 'o', label = 'line_1')
ax[1].plot(df.index.values, df['y'], color = 'orange', lw = 0, marker = 'o', label = 'line_2')
ax[0].legend(loc = 'upper right')
ax[1].legend(loc = 'lower center')
#%%
'''
add preprocessing
'''
#%%
'''
add label encoder first
add select k best
'''
le = LabelEncoder()
le_count = 0

for col in bank_df.columns:
    #if bank_df[col].dtype == 'object':
    le.fit(bank_df[col])
    bank_df[col] = le.transform(bank_df[col])
    le_count += 1

print('%d columns were converted.' % le_count)

#for comparison of the old data frame and the new one
print("PROCESSED DATA FRAME:")
print(df.head(3))
k_best = SelectKBest(score_func = f_classif, k = 12)
k_best.fit(bank_df, bank_df['amount'])
k_best.get_params()