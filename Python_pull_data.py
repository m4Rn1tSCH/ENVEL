#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:24:04 2020

@author: bill
"""

""" Pull data and determine amount saved"""

from psycopg2 import OperationalError
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

from Python_SQL_connection import execute_read_query, create_connection
import PostgreSQL_credentials as acc
from Python_spending_report_csv_function import spending_report as create_spending_report

def pull_df(rng=4, spending_report=False, plots=False):

    '''
    Parameters
    ----------
    rng : int, Random Seed for user picker. The default is 4.
    spending_report : bool, Save a spending report in directory if True. Default is False.
    plots : bool, Plots various graphs if True. Default is False.
    include_lag_features : include lag feature 'amount' to database with 3, 7, and 30 day rolls. Default is True
    Returns
    -------
    df.
    '''

    connection = create_connection(db_name=acc.YDB_name,
                                   db_user=acc.YDB_user,
                                   db_password=acc.YDB_password,
                                   db_host=acc.YDB_host,
                                   db_port=acc.YDB_port)

    # establish connection to get user IDs for all users in MA
    filter_query = f"SELECT unique_mem_id, state, city, zip_code, income_class, file_created_date FROM user_demographic WHERE state = 'MA'"
    transaction_query = execute_read_query(connection, filter_query)
    query_df = pd.DataFrame(transaction_query,
                            columns=['unique_mem_id', 'state', 'city', 'zip_code', 'income_class', 'file_created_date'])

    try:
        for i in pd.Series(query_df['unique_mem_id'].unique()).sample(n=1, random_state=rng):
            filter_query = f"SELECT * FROM bank_record WHERE unique_mem_id = '{i}'"
            transaction_query = execute_read_query(connection, filter_query)
            df = pd.DataFrame(transaction_query,
                                   columns=['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id',
                                   'amount', 'currency', 'description', 'transaction_date', 'post_date', 'transaction_base_type',
                                   'transaction_category_name', 'primary_merchant_name', 'secondary_merchant_name', 'city',
                                   'state', 'zip_code', 'transaction_origin', 'factual_category', 'factual_id', 'file_created_date',
                                   'optimized_transaction_date', 'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred',
                                   'swipe_date', 'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
                                   'account_type', 'account_source_type', 'account_score', 'user_score', 'lag', 'is_duplicate'])
            print(f"User {i} has {len(df)} transactions on record.")
            # all these columns are empty or almost empty and contain no viable information
            df = df.drop(columns=['secondary_merchant_name', 'swipe_date', 'update_type', 'is_outlier',
                                  'is_duplicate', 'change_source', 'lag', 'mcc_inferred', 'mcc_raw',
                                  'factual_id', 'factual_category', 'zip_code', 'yodlee_transaction_status',
                                  'file_created_date', 'panel_file_created_date', 'account_source_type',
                                  'account_type', 'account_score', 'user_score'], axis=1)
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        connection.rollback

    '''
    Plotting of various relations
    The Counter object keeps track of permutations in a dictionary which can then be read and
    used as labels
    '''
    if plots:
        # Pie chart States
        state_ct = Counter(list(df['state']))
        # The * operator can be used in conjunction with zip() to unzip the list.
        labels, values = zip(*state_ct.items())
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        fig1, ax = plt.subplots(figsize=(20, 12))
        ax.pie(values, labels=labels, autopct='%1.1f%%',
              shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')
        #ax.title('Transaction locations of user {df[unique_mem_id][0]}')
        ax.legend(loc='center right')
        plt.show()
        
        # Pie chart transaction type
        trans_ct = Counter(list(df['transaction_category_name']))
        # The * operator can be used in conjunction with zip() to unzip the list.
        labels_2, values_2 = zip(*trans_ct.items())
        #Pie chart, where the slices will be ordered and plotted counter-clockwise:
        fig1, ax = plt.subplots(figsize=(20, 12))
        ax.pie(values_2, labels=labels_2, autopct='%1.1f%%',
              shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')
        ax.title("Transaction categories")
        ax.legend(loc='center right')
        plt.show()

    '''
    Generate a spending report of the unaltered dataframe
    Use the datetime columns just defined
    This report measures either the sum or mean of transactions happening
    on various days of the week/or wihtin a week or a month  over the course of the year
    '''

    df['post_date'] = pd.to_datetime(df['post_date'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['optimized_transaction_date'] = pd.to_datetime(
        df['optimized_transaction_date'])

    # set optimized transaction_date as index for later
    df.set_index('optimized_transaction_date', drop=False, inplace=True)

    # generate the spending report with the above randomly picked user ID
    if spending_report:
        create_spending_report(df=df.copy())

    return df
#%%
df = pull_df(rng=9,
             spending_report=True,
             plots=False)

'''
columns help for reporting like weekly or monthly expenses and
improve prediction of re-occurring transactions
'''
for col in list(df):
    if df[col].dtype == 'datetime64[ns]':
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_week"] = df[col].dt.week
        df[f"{col}_weekday"] = df[col].dt.weekday

    '''
    POSTGRESQL COLUMNS - CLASSIFICATION OF TRANSACTIONS
    Following lists contains the categories to classify transactions either as expense or income
    names taken directly from the Yodlee dataset; can be appended at will
    '''
    #append these unique to DFs measuring expenses or income with their respective categories
    # card_inc = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts']
    # card_exp = ['Groceries', 'Automotive/Fuel', 'Home Improvement', 'Travel',
    #             'Restaurants', 'Healthcare/Medical', 'Credit Card Payments',
    #             'Electronics/General Merchandise', 'Entertainment/Recreation',
    #             'Postage/Shipping', 'Other Expenses', 'Personal/Family',
    #             'Service Charges/Fees', 'Services/Supplies', 'Utilities',
    #             'Office Expenses', 'Cable/Satellite/Telecom',
    #             'Subscriptions/Renewals', 'Insurance']
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

    #DF_CARD

    # transaction_class_card = pd.Series([], dtype = 'object')
    # for index, i in enumerate(df_card['transaction_category_name']):
    #     if i in card_inc:
    #         transaction_class_card[index] = "income"
    #     elif i in card_exp:
    #         transaction_class_card[index] = "expense"
    #     else:
    #         transaction_class_card[index] = "NOT_CLASSIFIED"
    # df_card.insert(loc = len(df_card.columns), column = "transaction_class", value = transaction_class_card)

    #DF_BANK

    transaction_class = pd.Series([], dtype = 'object')
    for index, i in enumerate(df['transaction_category_name']):
        if i in bank_inc:
            transaction_class[index] = "income"
        elif i in bank_exp:
            transaction_class[index] = "expense"
        else:
            transaction_class[index] = "NOT_CLASSIFIED"

    df = df.assign(transaction_class=transaction_class.values)
    #except:
        #print("column is already existing or another error")

    income_by_user = df.iloc[df['transaction_base_type' == 'income']].groupby('optimized_transaction_date_month')

    expenses_by_user = df.eq('expense').sum()

