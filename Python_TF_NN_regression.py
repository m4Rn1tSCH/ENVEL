# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:07:59 2020

@author: bill-
"""
#import required packages
import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np
from datetime import datetime as dt
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from flask import Flask

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column, data
from tensorflow.keras import Model, layers, regularizers

#IMPORTED CUSTOM FUNCTION
#generates a CSV for daily/weekly/monthly account throughput; expenses and income
from Python_spending_report_csv_function import spending_report
#contains the connection script
from Python_SQL_connection import execute_read_query, create_connection, close_connection
#contains all credentials
import PostgreSQL_credentials as acc
#csv export with optional append-mode
from Python_CSV_export_function import csv_export
#%%
#app = Flask('df_conversion')
#@app.route('/df_encoder')
def df_encoder(rng = 4):

    '''
    Parameters
    ----------
    rng : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    df.

    '''

    connection = create_connection(db_name = acc.YDB_name,
                                   db_user = acc.YDB_user,
                                   db_password = acc.YDB_password,
                                   db_host = acc.YDB_host,
                                   db_port = acc.YDB_port)

    #establish connection to get user IDs
    filter_query = f"SELECT unique_mem_id, state, city, zip_code, income_class, file_created_date FROM user_demographic WHERE state = 'MA'"
    transaction_query = execute_read_query(connection, filter_query)
    query_df = pd.DataFrame(transaction_query,
                            columns = ['unique_mem_id', 'state', 'city', 'zip_code', 'income_class', 'file_created_date'])

    #dateframe to gather MA bank data from one randomly chosen user
    #test user 1= 4
    #test user 2= 8
    rng = 4
    try:
        for i in pd.Series(query_df['unique_mem_id'].unique()).sample(n = 1, random_state = rng):
            print(i)
            filter_query = f"SELECT * FROM bank_record WHERE unique_mem_id = '{i}'"
            transaction_query = execute_read_query(connection, filter_query)
            df = pd.DataFrame(transaction_query,
                            columns = ['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id','amount',
                                       'currency', 'description', 'transaction_date', 'post_date',
                                       'transaction_base_type', 'transaction_category_name', 'primary_merchant_name',
                                       'secondary_merchant_name', 'city','state', 'zip_code', 'transaction_origin',
                                       'factual_category', 'factual_id', 'file_created_date', 'optimized_transaction_date',
                                       'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred', 'swipe_date',
                                       'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
                                       'account_type', 'account_source_type', 'account_score', 'user_score', 'lag', 'is_duplicate'])
            print(f"User {i} has {len(df)} transactions on record.")
            #all these columns are empty or almost empty and contain no viable information
            df = df.drop(columns = ['secondary_merchant_name',
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
    csv_export(df=df, file_name='bank_dataframe')

    '''
    Plotting of various relations
    The Counter object keeps track of permutations in a dictionary which can then be read and
    used as labels
    '''
    #Pie chart States - works
    state_ct = Counter(list(df['state']))
    #The * operator can be used in conjunction with zip() to unzip the list.
    labels, values = zip(*state_ct.items())
    #Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig1, ax = plt.subplots(figsize = (20, 12))
    ax.pie(values, labels = labels, autopct = '%1.1f%%',
            shadow = True, startangle = 90)
    #Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    #ax.title('Transaction locations of user {df[unique_mem_id][0]}')
    ax.legend(loc = 'center right')
    plt.show()

    #Pie chart transaction type -works
    trans_ct = Counter(list(df['transaction_category_name']))
    #The * operator can be used in conjunction with zip() to unzip the list.
    labels_2, values_2 = zip(*trans_ct.items())
    #Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig1, ax = plt.subplots(figsize = (20, 12))
    ax.pie(values_2, labels = labels_2, autopct = '%1.1f%%',
            shadow = True, startangle = 90)
    #Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    #ax.title('Transaction categories of user {df[unique_mem_id][0]}')
    ax.legend(loc = 'center right')
    plt.show()


    '''
    Generate a spending report of the unaltered dataframe
    Use the datetime columns just defined
    This report measures either the sum or mean of transactions happening
    on various days of the week/or wihtin a week or a month  over the course of the year
    '''
    #convert all date col from date to datetime objects
    #date objects will block Select K Best if not converted
    #first conversion from date to datetime objects; then conversion to unix
    df['post_date'] = pd.to_datetime(df['post_date'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['optimized_transaction_date'] = pd.to_datetime(df['optimized_transaction_date'])
    df['file_created_date'] = pd.to_datetime(df['file_created_date'])
    df['panel_file_created_date'] = pd.to_datetime(df['panel_file_created_date'])

    #set optimized transaction_date as index for later
    df.set_index('optimized_transaction_date', drop = False, inplace=True)
    #generate the spending report with a randomly picked user ID
    #when datetime columns are still datetime objects the spending report works
    '''
    Weekday legend
    Mon: 0
    Tue: 1
    Wed: 2
    Thu: 3
    Fri: 4
    '''
    spending_report(df = df.copy())

    '''
    After successfully loading the data, columns that are of no importance have been removed and missing values replaced
    Then the dataframe is ready to be encoded to get rid of all non-numerical data
    '''
    try:
        df['unique_mem_id'] = df['unique_mem_id'].astype('str', errors = 'ignore')
        df['unique_bank_account_id'] = df['unique_bank_account_id'].astype('str', errors = 'ignore')
        df['unique_bank_transaction_id'] = df['unique_bank_transaction_id'].astype('str', errors = 'ignore')
        df['amount'] = df['amount'].astype('float64')
        df['transaction_base_type'] = df['transaction_base_type'].replace(to_replace = ["debit", "credit"], value = [1, 0])
    except (TypeError, OSError, ValueError) as e:
        print("Problem with conversion:")
        print(e)

#attempt to convert date objects if they have no missing values; otherwise they are being dropped
    try:
        #conversion of dates to unix timestamps as numeric value (fl64)
        if df['post_date'].isnull().sum() == 0:
            df['post_date'] = df['post_date'].apply(lambda x: dt.timestamp(x))
        else:
            df = df.drop(columns = 'post_date', axis = 1)
            print("Column post_date dropped")

        if df['transaction_date'].isnull().sum() == 0:
            df['transaction_date'] = df['transaction_date'].apply(lambda x: dt.timestamp(x))
        else:
            df = df.drop(columns = 'transaction_date', axis = 1)
            print("Column transaction_date dropped")

        if df['optimized_transaction_date'].isnull().sum() == 0:
            df['optimized_transaction_date'] = df['optimized_transaction_date'].apply(lambda x: dt.timestamp(x))
        else:
            df = df.drop(columns = 'optimized_transaction_date', axis = 1)
            print("Column optimized_transaction_date dropped")

        if df['file_created_date'].isnull().sum() == 0:
            df['file_created_date'] = df['file_created_date'].apply(lambda x: dt.timestamp(x))
        else:
            df = df.drop(columns = 'file_created_date', axis = 1)
            print("Column file_created_date dropped")

        if df['panel_file_created_date'].isnull().sum() == 0:
            df['panel_file_created_date'] = df['panel_file_created_date'].apply(lambda x: dt.timestamp(x))
        else:
            df = df.drop(columns = 'panel_file_created_date', axis = 1)
            print("Column panel_file_created_date dropped")
    except (TypeError, OSError, ValueError) as e:
        print("Problem with conversion:")
        print(e)

    '''
    The columns PRIMARY_MERCHANT_NAME; CITY, STATE, DESCRIPTION, TRANSACTION_CATEGORY_NAME, CURRENCY
    are encoded manually and cleared of empty values
    '''
    #WORKS
    #encoding merchants
    UNKNOWN_TOKEN = '<unknown>'
    merchants = df['primary_merchant_name'].unique().astype('str').tolist()
    #a = pd.Series(['A', 'B', 'C', 'D', 'A'], dtype=str).unique().tolist()
    merchants.append(UNKNOWN_TOKEN)
    le = LabelEncoder()
    le.fit_transform(merchants)
    embedding_map_merchants = dict(zip(le.classes_, le.transform(le.classes_)))

    #APPLICATION TO OUR DATASET
    df['primary_merchant_name'] = df['primary_merchant_name'].apply(lambda x:
                                                                              x if x in embedding_map_merchants else UNKNOWN_TOKEN)
    df['primary_merchant_name'] = df['primary_merchant_name'].map(lambda x:
                                                                            le.transform([x])[0] if type(x)==str else x)

    #encoding cities
    UNKNOWN_TOKEN = '<unknown>'
    cities = df['city'].unique().astype('str').tolist()
    cities.append(UNKNOWN_TOKEN)
    le_2 = LabelEncoder()
    le_2.fit_transform(cities)
    embedding_map_cities = dict(zip(le_2.classes_, le_2.transform(le_2.classes_)))

    #APPLICATION TO OUR DATASET
    df['city'] = df['city'].apply(lambda x: x if x in embedding_map_cities else UNKNOWN_TOKEN)
    df['city'] = df['city'].map(lambda x: le_2.transform([x])[0] if type(x)==str else x)

    #encoding states
    states = df['state'].unique().astype('str').tolist()
    states.append(UNKNOWN_TOKEN)
    le_3 = LabelEncoder()
    le_3.fit_transform(states)
    embedding_map_states = dict(zip(le_3.classes_, le_3.transform(le_3.classes_)))

    #APPLICATION TO OUR DATASET
    df['state'] = df['state'].apply(lambda x: x if x in embedding_map_states else UNKNOWN_TOKEN)
    df['state'] = df['state'].map(lambda x: le_3.transform([x])[0] if type(x)==str else x)

    #encoding descriptions
    desc = df['description'].unique().astype('str').tolist()
    desc.append(UNKNOWN_TOKEN)
    le_4 = LabelEncoder()
    le_4.fit_transform(desc)
    embedding_map_desc = dict(zip(le_4.classes_, le_4.transform(le_4.classes_)))

    #APPLICATION TO OUR DATASET
    df['description'] = df['description'].apply(lambda x: x if x in embedding_map_desc else UNKNOWN_TOKEN)
    df['description'] = df['description'].map(lambda x: le_4.transform([x])[0] if type(x)==str else x)

    #encoding descriptions
    desc = df['transaction_category_name'].unique().astype('str').tolist()
    desc.append(UNKNOWN_TOKEN)
    le_5 = LabelEncoder()
    le_5.fit_transform(desc)
    embedding_map_tcat = dict(zip(le_5.classes_, le_5.transform(le_5.classes_)))

    #APPLICATION TO OUR DATASET
    df['transaction_category_name'] = df['transaction_category_name'].apply(lambda x:
                                                                                      x if x in embedding_map_tcat else UNKNOWN_TOKEN)
    df['transaction_category_name'] = df['transaction_category_name'].map(lambda x:
                                                                                    le_5.transform([x])[0] if type(x)==str else x)

    #encoding transaction origin
    desc = df['transaction_origin'].unique().astype('str').tolist()
    desc.append(UNKNOWN_TOKEN)
    le_6 = LabelEncoder()
    le_6.fit_transform(desc)
    embedding_map_tori = dict(zip(le_6.classes_, le_6.transform(le_6.classes_)))

    #APPLICATION TO OUR DATASET
    df['transaction_origin'] = df['transaction_origin'].apply(lambda x:
                                                                        x if x in embedding_map_tori else UNKNOWN_TOKEN)
    df['transaction_origin'] = df['transaction_origin'].map(lambda x:
                                                                      le_6.transform([x])[0] if type(x)==str else x)

    #encoding currency if there is more than one in use
    try:
        if len(df['currency'].value_counts()) == 1:
            df = df.drop(columns = ['currency'], axis = 1)
        elif len(df['currency'].value_counts()) > 1:
            #encoding merchants
            UNKNOWN_TOKEN = '<unknown>'
            currencies = df['currency'].unique().astype('str').tolist()
            #a = pd.Series(['A', 'B', 'C', 'D', 'A'], dtype=str).unique().tolist()
            currencies.append(UNKNOWN_TOKEN)
            le_7 = LabelEncoder()
            le_7.fit_transform(merchants)
            embedding_map_currency = dict(zip(le_7.classes_, le_7.transform(le_7.classes_)))
            df['currency'] = df['currency'].apply(lambda x:
                                                            x if x in embedding_map_currency else UNKNOWN_TOKEN)
            df['currency'] = df['currency'].map(lambda x:
                                                          le_7.transform([x])[0] if type(x)==str else x)
    except:
        print("Column currency was not converted.")
        pass

    '''
    IMPORTANT
    The lagging features produce NaN for the first two rows due to unavailability
    of values
    NaNs need to be dropped to make scaling and selection of features working
    '''
    # TEMPORARY SOLUTION; Add date columns for more accurate overview
    # Set up a rolling time window that is calculating lagging cumulative spending
    # '''
    # for col in list(df):
    #     if df[col].dtype == 'datetime64[ns]':
    #         df[f"{col}_month"] = df[col].dt.month
    #         df[f"{col}_week"] = df[col].dt.week
    #         df[f"{col}_weekday"] = df[col].dt.weekday

    #FEATURE ENGINEERING
    #typical engineered features based on lagging metrics
    #mean + stdev of past 3d/7d/30d/ + rolling volume
    date_index = df.index.values
    df.reset_index(drop = True, inplace = True)
    #pick lag features to iterate through and calculate features
    lag_features = ["amount"]
    #set up time frames; how many days/months back/forth
    t1 = 3
    t2 = 7
    t3 = 30
    #rolling values for all columns ready to be processed
    df_rolled_3d = df[lag_features].rolling(window = t1, min_periods = 0)
    df_rolled_7d = df[lag_features].rolling(window = t2, min_periods = 0)
    df_rolled_30d = df[lag_features].rolling(window = t3, min_periods = 0)

    #calculate the mean with a shifting time window
    df_mean_3d = df_rolled_3d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_mean_7d = df_rolled_7d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_mean_30d = df_rolled_30d.mean().shift(periods = 1).reset_index().astype(np.float32)

    #calculate the std dev with a shifting time window
    df_std_3d = df_rolled_3d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_std_7d = df_rolled_7d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_std_30d = df_rolled_30d.std().shift(periods = 1).reset_index().astype(np.float32)

    for feature in lag_features:
        df[f"{feature}_mean_lag{t1}"] = df_mean_3d[feature]
        df[f"{feature}_mean_lag{t2}"] = df_mean_7d[feature]
        df[f"{feature}_mean_lag{t3}"] = df_mean_30d[feature]

        df[f"{feature}_std_lag{t1}"] = df_std_3d[feature]
        df[f"{feature}_std_lag{t2}"] = df_std_7d[feature]
        df[f"{feature}_std_lag{t3}"] = df_std_30d[feature]

    df.set_index(date_index, drop = False, inplace=True)
    df = df.dropna()
    # drop user IDs to avoid overfitting with useless information
    df = df.drop(['unique_mem_id',
                            'unique_bank_account_id',
                            'unique_bank_transaction_id'], axis = 1)

    # seaborn plots
    ax_desc = df['description'].astype('int64', errors='ignore')
    ax_amount = df['amount'].astype('int64',errors='ignore')
    sns.pairplot(df)
    sns.boxplot(x=ax_desc, y=ax_amount)
    sns.heatmap(df)

    return df
#%%
'''
                Tensorflow Regression
'''
#app = Flask('nn_reg')
#@app.route('/nn_tf')
# python automatically passes self to a function; can be cause for 1 arg given even though function takes 0 arguments
#def nn_regression(self):

print("tensorflow regression running...")
bank_df = df_encoder(rng=4)
dataset = bank_df.copy()
print(dataset.head())
print(dataset.tail())
print(dataset.isna().sum())
sns.pairplot(bank_df[['amount', 'amount_mean_lag7', 'amount_std_lag7']])
#%%
# setting label and features (the df itself here)
model_label = dataset.pop('amount_mean_lag7')
model_label.astype('int64')

# EAGER EXECUTION NEEDS TO BE ENABLED HERE
# features and model labels passed as tuple
tensor_ds = tf.data.Dataset.from_tensor_slices((dataset.values, model_label.values))
for feat, targ in tensor_ds.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))

train_dataset = tensor_ds.shuffle(len(bank_df)).batch(2)
#%%
#OPTIONAL WAY TO FEED DATA
##########################################
# ALTERNATIVE TO FEED FEATURES TO THE MODEL AS DICTIONARY CONSISTING OF DF KEYS
# inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
# x = tf.stack(list(inputs.values()), axis=-1)

# x = tf.keras.layers.Dense(10, activation='relu')(x)
# output = tf.keras.layers.Dense(1)(x)

# model_func = tf.keras.Model(inputs=inputs, outputs=output)

# model_func.compile(optimizer='adam',
#                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                     metrics=['mse', 'mae'])

# dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)
# for dict_slice in dict_slices.take(1):
#   print (dict_slice)
# model_func.fit(dict_slices, epochs=15)
###########################################
#%%
# if numpy arrays are given the first layers needs to be layers.Flatten and specify the quadratic input shape
# REGULARIZATION: add dropout or regularizers in each layer
# combined reg + dropout produces best results
# Dropout layers sets variables to zero in randomized patterns
# when adding weight relugarizer - monitor bin cross entropy directly
def compile_model():
    model = tf.keras.Sequential([
    # initial layer is input layers
    tf.keras.layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    # final layers is output layers
    tf.keras.layers.Dense(1)
    ])
    # gradients / running average; optimizes stochastic gradient descent
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer,
                  # mse is a for a loss function in regressions
                  loss='mse',
                  # mae is an evaluation metric for regression problems (needs list)
                  metrics=['mae'])
    return model
#%%
model = compile_model()
#model.fit(train_dataset, epochs=15)

EPOCHS = 50
# when dataset given; only ds needed (no y needed)
# when separate tensors given; arg x and y needed
history = model.fit(train_dataset, epochs=EPOCHS, verbose=2)

# regularizer_histories['reg_and_dropout'] = compile_model(combined_model, "regularizers/combined")
# plt.plot(history)


