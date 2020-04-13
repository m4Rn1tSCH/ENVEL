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
connection = create_connection(db_name = acc.YDB_name,
                               db_user = acc.YDB_user,
                               db_password = acc.YDB_password,
                               db_host = acc.YDB_host,
                               db_port = acc.YDB_port)
#%%
#estbalish connection to get user IDs
filter_query = f"SELECT unique_mem_id, state, city, zip_code, income_class, file_created_date FROM user_demographic WHERE state = 'MA'"
transaction_query = execute_read_query(connection, filter_query)
query_df = pd.DataFrame(transaction_query,
                        columns = ['unique_mem_id', 'state', 'city', 'zip_code', 'income_class', 'file_created_date'])
#%%
#dateframe to gather MA bank data from one randomly chosen user
try:
    for i in query_df['unique_mem_id'].sample(n = 1, random_state = 2):
        print(i)
        filter_query = f"SELECT * FROM bank_record WHERE unique_mem_id = '{i}'"
        transaction_query = execute_read_query(connection, filter_query)
        bank_df = pd.DataFrame(transaction_query,
                        columns = ['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id','amount',
                                   'currency', 'description', 'transaction_date', 'post_date',
                                   'transaction_base_type', 'transaction_category_name', 'primary_merchant_name',
                                   'secondary_merchant_name', 'city','state', 'zip_code', 'transaction_origin',
                                   'factual_category', 'factual_id', 'file_created_date', 'optimized_transaction_date',
                                   'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred', 'swipe_date',
                                   'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
                                   'account_type', 'account_source_type', 'account_score', 'user_score', 'lag', 'is_duplicate'])
        print(f"User {i} has {len(bank_df)} transactions on record.")
        #all these columns are empty or almost empty and contain no viable information
        bank_df = bank_df.drop(columns = ['secondary_merchant_name',
                                          'swipe_date',
                                          'update_type',
                                          'is_outlier' ,
                                          'is_duplicate',
                                          'change_source',
                                          'lag',
                                          'mcc_inferred',
                                          'mcc_raw',
                                          'factual_id',
                                          'factual_category',
                                          'zip_code',
                                          'yodlee_transaction_status'], axis = 1)
except OperationalError as e:
        print(f"The error '{e}' occurred")
        connection.rollback
#%%
#Plot template
# fig, ax = plt.subplots(2, 1, figsize = (25, 25))
# ax[0].plot(df.index.values, df['x'], color = 'green', lw = 4, ls = '-.', marker = 'o', label = 'line_1')
# ax[1].plot(df.index.values, df['y'], color = 'orange', lw = 0, marker = 'o', label = 'line_2')
# ax[0].legend(loc = 'upper right')
# ax[1].legend(loc = 'lower center')

#Pie chart template
# labels, values = zip(*tx_types.items())
# # Pie chart, where the slices will be ordered and plotted counter-clockwise:
# fig1, ax1 = plt.subplots()
# ax1.pie(values, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.show()

#Boxplot template
# cat_var = ["type", "check", "institutionName", "feeDescription", "Student", "isCredit", "CS_FICO_str"]
# quant_var = ["Age", "amount"]
# for c_var in cat_var:
#     for q_var in quant_var:
#         df.boxplot(column=q_var, by=c_var)
#         plt.xticks([])
#%%
'''
After successfully loading the data, columns that are of no importance will be removed and missing values replaced
Then the dataframe is ready to be encoded to get rid of all non-numerical data
add preprocessing
'''
# print(bank_df[bank_df['city'].isnull()])
# #Then for remove all not numeric values use to_numeric with parameetr errors='coerce' - it replace non numeric to NaNs:
# bank_df['x'] = pd.to_numeric(bank_df['x'], errors='coerce')
# #And for remove all rows with NaNs in column x use dropna:
# bank_df = bank_df.dropna(subset=['x'])
# #Last convert values to ints:
# bank_df['x'] = bank_df['x'].astype(int)

# try:
#     bank_df['city'].replace("None", "UNKNOWN")
#     bank_df['state'].replace("None", "UNKNOWN")
# #    bank_df.fillna(value = 'unknown')
# except TypeError as e:
#     print(e)
#     pass

bank_df['primary_merchant_name'].fillna(value = 'unknown')

#THIS MIGHT RUIN THE DATA; SINCE THE STATE REFERS TO THE TANSACTION LOCATION NOT TE ORIGIN OF THE USER
#bank_df['state'].fillna(value = 'MA')
#bank_df['city'].fillna(value = 'unknown')

#bank_df['factual_category'].fillna(value = 'unknown')
#bank_df['factual_id'].fillna(value = 'unknown')

#prepare numeric and string columns
bank_df['unique_bank_account_id'] = bank_df['unique_bank_account_id'].astype('str', errors = 'ignore')
bank_df['unique_bank_transaction_id'] = bank_df['unique_bank_transaction_id'].astype('str', errors = 'ignore')
bank_df['amount'] = bank_df['amount'].astype('float64')
bank_df['currency'].astype('str', errors = 'ignore')
bank_df['description'] = bank_df['description'].astype('str')
bank_df['transaction_base_type'] = bank_df['transaction_base_type'].astype('str')
bank_df['transaction_category_name'].astype('str')
bank_df['primary_merchant_name'].astype('str')
bank_df['city'].astype('str')
bank_df['state'].astype('str')
#bank_df['zip_code'].astype('str')
bank_df['transaction_origin'].astype('str')

#concert all datetime columns
bank_df['transaction_date'] = pd.to_datetime(bank_df['transaction_date'])
bank_df['optimized_transaction_date'] = pd.to_datetime(bank_df['optimized_transaction_date'])
bank_df['file_created_date'] = pd.to_datetime(bank_df['file_created_date'])
bank_df['panel_file_created_date'] = pd.to_datetime(bank_df['panel_file_created_date'])
#%%
'''
add label encoder first
add select K BEST
'''
#WORKS
#encoding merchants
UNKNOWN_TOKEN = '<unknown>'
merchants = bank_df['primary_merchant_name'].unique().astype('str').tolist()
#a = pd.Series(['A', 'B', 'C', 'D', 'A'], dtype=str).unique().tolist()
merchants.append(UNKNOWN_TOKEN)
le = LabelEncoder()
le.fit_transform(merchants)
embedding_map_merchants = dict(zip(le.classes_, le.transform(le.classes_)))

#APPLICATION TO OUR DATASET
bank_df['primary_merchant_name'] = bank_df['primary_merchant_name'].apply(lambda x: x if x in embedding_map_merchants else UNKNOWN_TOKEN)
#le.transform(bank_df)
bank_df['primary_merchant_name'] = bank_df['primary_merchant_name'].map(lambda x: le.transform([x])[0] if type(x)==str else x)

#encoding cities
UNKNOWN_TOKEN = '<unknown>'
cities = bank_df['city'].unique().astype('str').tolist()
cities.append(UNKNOWN_TOKEN)
le_2 = LabelEncoder()
le_2.fit_transform(cities)
embedding_map_cities = dict(zip(le_2.classes_, le_2.transform(le_2.classes_)))

#APPLICATION TO OUR DATASET
bank_df['city'] = bank_df['city'].apply(lambda x: x if x in embedding_map_cities else UNKNOWN_TOKEN)
#le_2.transform(bank_df)
bank_df['city'] = bank_df['city'].map(lambda x: le_2.transform([x])[0] if type(x)==str else x)


#encoding states
#UNKNOWN_TOKEN = '<unknown>'
states = bank_df['state'].unique().astype('str').tolist()
states.append(UNKNOWN_TOKEN)
le_3 = LabelEncoder()
le_3.fit_transform(states)
embedding_map_states = dict(zip(le_3.classes_, le_3.transform(le.classes_)))

#APPLICATION TO OUR DATASET
bank_df['state'] = bank_df['state'].apply(lambda x: x if x in embedding_map_states else UNKNOWN_TOKEN)
#le_3.transform(bank_df)
bank_df['state'] = bank_df['state'].map(lambda x: le_3.transform([x])[0] if type(x)==str else x)
#%%
y = bank_df['primary_merchant_name']
X = bank_df.drop(columns = 'primary_merchant_name', axis = 1)
k_best = SelectKBest(score_func = f_classif, k = 10)
k_best.fit(X, y)
k_best.get_params()

# isCredit_num = [1 if x == 'Y' else 0 for x in isCredits]
# np.corrcoef(np.array(isCredit_num), amounts)