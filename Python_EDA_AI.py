#future needs to be run first
#eager execution needs to be run right after the TF instantiation to avoid errors
from __future__ import absolute_import, division, print_function, unicode_literals
import functools

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:51:04 2020

@author: bill-
"""
'''
EDA module for various Yodlee dataframes
FIRST STAGE: retrieve the user ID dataframe with all user IDs with given filter
    dataframe called bank_df is being generated in the current work directory as CSV
SECOND STAGE: randomly pick a user ID; encode thoroughly and yield the df
THIRD STAGE: encode all columns to numerical values and store corresponding dictionaries
'''

#load packages
import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np
from datetime import datetime as dt
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

from sklearn.feature_selection import SelectKBest , chi2, f_classif, RFE, RFECV
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline


import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation



import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column
from tensorflow.keras import layers

#imported custom function
#generates a CSV for daily/weekly/monthly account throughput; expenses and income
from Python_spending_report_csv_function import spending_report
#contains the connection script
from Python_SQL_connection import execute_read_query, create_connection, close_connection
#contains all credentials
import PostgreSQL_credentials as acc
#loads flask into the environment variables
from flask import Flask
#csv export with optional append-mode
from Python_CSV_export_function import csv_export
#%%
#set up flask first as environment variable and start with the command console
    #CONNECTION TO FLASK/SQL
app = Flask(__name__)

##put address here
#function can be bound to the script by adding a new URL
#e.g. route('/start') would then start the entire function that follows
#same can be split up
@app.route('/encode')

def df_encoder(rng = 4):
    '''

    Parameters
    ----------
    rng : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    bank_df.

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
    #csv_export(df=bank_df, file_name='bank_dataframe')

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

    # #Pie chart States
    # state_ct = Counter(list(bank_df['state']))
    # #asterisk look up, what is that?
    # labels, values = zip(*state_ct.items())
    # #Pie chart, where the slices will be ordered and plotted counter-clockwise:
    # fig1, ax1 = plt.subplots()
    # ax1.pie(values, labels = labels, autopct = '%1.1f%%',
    #         shadow = True, startangle = 90)
    # #Equal aspect ratio ensures that pie is drawn as a circle.
    # ax1.axis('equal')
    # plt.show()

    #Boxplot template
    # cat_var = ["type", "check", "institutionName", "feeDescription", "Student", "isCredit", "CS_FICO_str"]
    # quant_var = ["Age", "amount"]
    # for c_var in cat_var:
    #     for q_var in quant_var:
    #         df.boxplot(column=q_var, by=c_var)
    #         plt.xticks([])

    '''
    Plotting of various relations
    The Counter object keeps track of permutations in a dictionary which can then be read and
    used as labels
    '''
    #Pie chart States - works
    state_ct = Counter(list(bank_df['state']))
    #The * operator can be used in conjunction with zip() to unzip the list.
    labels, values = zip(*state_ct.items())
    #Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig1, ax = plt.subplots(figsize = (20, 12))
    ax.pie(values, labels = labels, autopct = '%1.1f%%',
            shadow = True, startangle = 90)
    #Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    #ax.title('Transaction locations of user {bank_df[unique_mem_id][0]}')
    ax.legend(loc = 'center right')
    plt.show()

    #Pie chart transaction type -works
    trans_ct = Counter(list(bank_df['transaction_category_name']))
    #The * operator can be used in conjunction with zip() to unzip the list.
    labels_2, values_2 = zip(*trans_ct.items())
    #Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig1, ax = plt.subplots(figsize = (20, 12))
    ax.pie(values_2, labels = labels_2, autopct = '%1.1f%%',
            shadow = True, startangle = 90)
    #Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    #ax.title('Transaction categories of user {bank_df[unique_mem_id][0]}')
    ax.legend(loc = 'center right')
    plt.show()

    #seaborn plots
    # ax_desc = bank_df['description'].astype('int64', errors='ignore')
    # ax_amount = bank_df['amount'].astype('int64',errors='ignore')
    # sns.pairplot(bank_df)
    # sns.boxplot(x=ax_desc, y=ax_amount)
    # sns.heatmap(bank_df)

#BUGGED
    #Boxplot template
    # cat_var = ["unique_mem_id", "primary_merchant_name"]
    # quant_var = ["amount"]
    # for c_var in cat_var:
    #     for q_var in quant_var:
    #         bank_df.boxplot(column=q_var, by=c_var)
    #         plt.xticks([])

    '''
    Generate a spending report of the unaltered dataframe
    Use the datetime columns just defined
    This report measures either the sum or mean of transactions happening
    on various days of the week/or wihtin a week or a month  over the course of the year
    '''
    #convert all date col from date to datetime objects
    #date objects will block Select K Best if not converted
    #first conversion from date to datetime objects; then conversion to unix
    bank_df['post_date'] = pd.to_datetime(bank_df['post_date'])
    bank_df['transaction_date'] = pd.to_datetime(bank_df['transaction_date'])
    bank_df['optimized_transaction_date'] = pd.to_datetime(bank_df['optimized_transaction_date'])
    bank_df['file_created_date'] = pd.to_datetime(bank_df['file_created_date'])
    bank_df['panel_file_created_date'] = pd.to_datetime(bank_df['panel_file_created_date'])

    #set optimized transaction_date as index for later
    bank_df.set_index('optimized_transaction_date', drop = False, inplace=True)
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
    #spending_report(df = bank_df.copy())

    '''
    After successfully loading the data, columns that are of no importance have been removed and missing values replaced
    Then the dataframe is ready to be encoded to get rid of all non-numerical data
    '''
    try:
        bank_df['unique_mem_id'] = bank_df['unique_mem_id'].astype('str', errors = 'ignore')
        bank_df['unique_bank_account_id'] = bank_df['unique_bank_account_id'].astype('str', errors = 'ignore')
        bank_df['unique_bank_transaction_id'] = bank_df['unique_bank_transaction_id'].astype('str', errors = 'ignore')
        bank_df['amount'] = bank_df['amount'].astype('float64')
        bank_df['transaction_base_type'] = bank_df['transaction_base_type'].replace(to_replace = ["debit", "credit"], value = [1, 0])
    except (TypeError, OSError, ValueError) as e:
        print("Problem with conversion:")
        print(e)

#attempt to convert date objects if they have no missing values; otherwise they are being dropped
    try:
        #conversion of dates to unix timestamps as numeric value (fl64)
        if bank_df['post_date'].isnull().sum() == 0:
            bank_df['post_date'] = bank_df['post_date'].apply(lambda x: dt.timestamp(x))
        else:
            bank_df = bank_df.drop(columns = 'post_date', axis = 1)
            print("Column post_date dropped")

        if bank_df['transaction_date'].isnull().sum() == 0:
            bank_df['transaction_date'] = bank_df['transaction_date'].apply(lambda x: dt.timestamp(x))
        else:
            bank_df = bank_df.drop(columns = 'transaction_date', axis = 1)
            print("Column transaction_date dropped")

        if bank_df['optimized_transaction_date'].isnull().sum() == 0:
            bank_df['optimized_transaction_date'] = bank_df['optimized_transaction_date'].apply(lambda x: dt.timestamp(x))
        else:
            bank_df = bank_df.drop(columns = 'optimized_transaction_date', axis = 1)
            print("Column optimized_transaction_date dropped")

        if bank_df['file_created_date'].isnull().sum() == 0:
            bank_df['file_created_date'] = bank_df['file_created_date'].apply(lambda x: dt.timestamp(x))
        else:
            bank_df = bank_df.drop(columns = 'file_created_date', axis = 1)
            print("Column file_created_date dropped")

        if bank_df['panel_file_created_date'].isnull().sum() == 0:
            bank_df['panel_file_created_date'] = bank_df['panel_file_created_date'].apply(lambda x: dt.timestamp(x))
        else:
            bank_df = bank_df.drop(columns = 'panel_file_created_date', axis = 1)
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
    merchants = bank_df['primary_merchant_name'].unique().astype('str').tolist()
    #a = pd.Series(['A', 'B', 'C', 'D', 'A'], dtype=str).unique().tolist()
    merchants.append(UNKNOWN_TOKEN)
    le = LabelEncoder()
    le.fit_transform(merchants)
    embedding_map_merchants = dict(zip(le.classes_, le.transform(le.classes_)))

    #APPLICATION TO OUR DATASET
    bank_df['primary_merchant_name'] = bank_df['primary_merchant_name'].apply(lambda x:
                                                                              x if x in embedding_map_merchants else UNKNOWN_TOKEN)
    bank_df['primary_merchant_name'] = bank_df['primary_merchant_name'].map(lambda x:
                                                                            le.transform([x])[0] if type(x)==str else x)

    #encoding cities
    UNKNOWN_TOKEN = '<unknown>'
    cities = bank_df['city'].unique().astype('str').tolist()
    cities.append(UNKNOWN_TOKEN)
    le_2 = LabelEncoder()
    le_2.fit_transform(cities)
    embedding_map_cities = dict(zip(le_2.classes_, le_2.transform(le_2.classes_)))

    #APPLICATION TO OUR DATASET
    bank_df['city'] = bank_df['city'].apply(lambda x: x if x in embedding_map_cities else UNKNOWN_TOKEN)
    bank_df['city'] = bank_df['city'].map(lambda x: le_2.transform([x])[0] if type(x)==str else x)

    #encoding states
    states = bank_df['state'].unique().astype('str').tolist()
    states.append(UNKNOWN_TOKEN)
    le_3 = LabelEncoder()
    le_3.fit_transform(states)
    embedding_map_states = dict(zip(le_3.classes_, le_3.transform(le_3.classes_)))

    #APPLICATION TO OUR DATASET
    bank_df['state'] = bank_df['state'].apply(lambda x: x if x in embedding_map_states else UNKNOWN_TOKEN)
    bank_df['state'] = bank_df['state'].map(lambda x: le_3.transform([x])[0] if type(x)==str else x)

    #encoding descriptions
    desc = bank_df['description'].unique().astype('str').tolist()
    desc.append(UNKNOWN_TOKEN)
    le_4 = LabelEncoder()
    le_4.fit_transform(desc)
    embedding_map_desc = dict(zip(le_4.classes_, le_4.transform(le_4.classes_)))

    #APPLICATION TO OUR DATASET
    bank_df['description'] = bank_df['description'].apply(lambda x: x if x in embedding_map_desc else UNKNOWN_TOKEN)
    bank_df['description'] = bank_df['description'].map(lambda x: le_4.transform([x])[0] if type(x)==str else x)

    #encoding descriptions
    desc = bank_df['transaction_category_name'].unique().astype('str').tolist()
    desc.append(UNKNOWN_TOKEN)
    le_5 = LabelEncoder()
    le_5.fit_transform(desc)
    embedding_map_tcat = dict(zip(le_5.classes_, le_5.transform(le_5.classes_)))

    #APPLICATION TO OUR DATASET
    bank_df['transaction_category_name'] = bank_df['transaction_category_name'].apply(lambda x:
                                                                                      x if x in embedding_map_tcat else UNKNOWN_TOKEN)
    bank_df['transaction_category_name'] = bank_df['transaction_category_name'].map(lambda x:
                                                                                    le_5.transform([x])[0] if type(x)==str else x)

    #encoding transaction origin
    desc = bank_df['transaction_origin'].unique().astype('str').tolist()
    desc.append(UNKNOWN_TOKEN)
    le_6 = LabelEncoder()
    le_6.fit_transform(desc)
    embedding_map_tori = dict(zip(le_6.classes_, le_6.transform(le_6.classes_)))

    #APPLICATION TO OUR DATASET
    bank_df['transaction_origin'] = bank_df['transaction_origin'].apply(lambda x:
                                                                        x if x in embedding_map_tori else UNKNOWN_TOKEN)
    bank_df['transaction_origin'] = bank_df['transaction_origin'].map(lambda x:
                                                                      le_6.transform([x])[0] if type(x)==str else x)

    #encoding currency if there is more than one in use
    try:
        if len(bank_df['currency'].value_counts()) == 1:
            bank_df = bank_df.drop(columns = ['currency'], axis = 1)
        elif len(bank_df['currency'].value_counts()) > 1:
            #encoding merchants
            UNKNOWN_TOKEN = '<unknown>'
            currencies = bank_df['currency'].unique().astype('str').tolist()
            #a = pd.Series(['A', 'B', 'C', 'D', 'A'], dtype=str).unique().tolist()
            currencies.append(UNKNOWN_TOKEN)
            le_7 = LabelEncoder()
            le_7.fit_transform(merchants)
            embedding_map_currency = dict(zip(le_7.classes_, le_7.transform(le_7.classes_)))
            bank_df['currency'] = bank_df['currency'].apply(lambda x:
                                                            x if x in embedding_map_currency else UNKNOWN_TOKEN)
            bank_df['currency'] = bank_df['currency'].map(lambda x:
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
    # for col in list(bank_df):
    #     if bank_df[col].dtype == 'datetime64[ns]':
    #         bank_df[f"{col}_month"] = bank_df[col].dt.month
    #         bank_df[f"{col}_week"] = bank_df[col].dt.week
    #         bank_df[f"{col}_weekday"] = bank_df[col].dt.weekday

    #FEATURE ENGINEERING
    #typical engineered features based on lagging metrics
    #mean + stdev of past 3d/7d/30d/ + rolling volume
    date_index = bank_df.index.values
    bank_df.reset_index(drop = True, inplace = True)
    #pick lag features to iterate through and calculate features
    lag_features = ["amount"]
    #set up time frames; how many days/months back/forth
    t1 = 3
    t2 = 7
    t3 = 30
    #rolling values for all columns ready to be processed
    bank_df_rolled_3d = bank_df[lag_features].rolling(window = t1, min_periods = 0)
    bank_df_rolled_7d = bank_df[lag_features].rolling(window = t2, min_periods = 0)
    bank_df_rolled_30d = bank_df[lag_features].rolling(window = t3, min_periods = 0)

    #calculate the mean with a shifting time window
    bank_df_mean_3d = bank_df_rolled_3d.mean().shift(periods = 1).reset_index().astype(np.float32)
    bank_df_mean_7d = bank_df_rolled_7d.mean().shift(periods = 1).reset_index().astype(np.float32)
    bank_df_mean_30d = bank_df_rolled_30d.mean().shift(periods = 1).reset_index().astype(np.float32)

    #calculate the std dev with a shifting time window
    bank_df_std_3d = bank_df_rolled_3d.std().shift(periods = 1).reset_index().astype(np.float32)
    bank_df_std_7d = bank_df_rolled_7d.std().shift(periods = 1).reset_index().astype(np.float32)
    bank_df_std_30d = bank_df_rolled_30d.std().shift(periods = 1).reset_index().astype(np.float32)

    for feature in lag_features:
        bank_df[f"{feature}_mean_lag{t1}"] = bank_df_mean_3d[feature]
        bank_df[f"{feature}_mean_lag{t2}"] = bank_df_mean_7d[feature]
        bank_df[f"{feature}_mean_lag{t3}"] = bank_df_mean_30d[feature]

        bank_df[f"{feature}_std_lag{t1}"] = bank_df_std_3d[feature]
        bank_df[f"{feature}_std_lag{t2}"] = bank_df_std_7d[feature]
        bank_df[f"{feature}_std_lag{t3}"] = bank_df_std_30d[feature]

    bank_df.set_index(date_index, drop = False, inplace=True)
    bank_df = bank_df.dropna()
    #drop user IDs to avoid overfitting with useless information
    bank_df = bank_df.drop(['unique_mem_id',
                            'unique_bank_account_id',
                            'unique_bank_transaction_id'], axis = 1)
    return bank_df
    #%%
@app.route('/encode')
def split_data():
    ###################SPLITTING UP THE DATA###########################
    #drop target variable in feature df
    ####
    model_features = bank_df.drop(['amount_mean_lag7'], axis = 1)
    #On some occasions the label needs to be a 1d array;
    #then the double square brackets (slicing it as a new dataframe) break the pipeline
    model_label = bank_df['amount_mean_lag7']
    ####
    if model_label.dtype == 'float32':
        model_label = model_label.astype('int32')
    elif model_label.dtype == 'float64':
        model_label = model_label.astype('int64')
    else:
        print("model label has unsuitable data type!")


    #stratify needs to be applied when the labels are imbalanced and mainly just one/two permutation
    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        random_state = 7,
                                                        shuffle = True,
                                                        test_size = 0.4)

    #create a validation set from the training set
    print(f"Shape of the split training data set X_train:{X_train.shape}")
    print(f"Shape of the split training data set X_test: {X_test.shape}")
    print(f"Shape of the split training data set y_train: {y_train.shape}")
    print(f"Shape of the split training data set y_test: {y_test.shape}")

    #STD SCALING - does not work yet
    #fit the scaler to the training data first
    #standard scaler works only with maximum 2 dimensions
    scaler = StandardScaler(copy = True, with_mean = True, with_std = True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    #transform test data with the object learned from the training data
    X_test_scaled = scaler.transform(X_test)
    scaler_mean = scaler.mean_
    stadard_scale = scaler.scale_

    #MINMAX SCALING - works with Select K Best
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)

    X_test_minmax = min_max_scaler.transform(X_test)
    minmax_scale = min_max_scaler.scale_
    min_max_minimum = min_max_scaler.min_

    #Principal Component Reduction
    #first scale
    #then reduce
    #keep the most important features of the data
    pca = PCA(n_components = int(len(bank_df.columns) / 2))
    #fit PCA model to breast cancer data
    pca.fit(X_train_scaled)
    #transform data onto the first two principal components
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print("Original shape: {}".format(str(X_train_scaled.shape)))
    print("Reduced shape: {}".format(str(X_train_pca.shape)))

    '''
                Plotting of PCA/ Cluster Pairs
    '''
    #Kmeans clusters to categorize groups WITH SCALED DATA
    #determine number of groups needed or desired for
    kmeans = KMeans(n_clusters = 10, random_state = 10)
    train_clusters = kmeans.fit(X_train_scaled)

    kmeans = KMeans(n_clusters = 10, random_state = 10)
    test_clusters = kmeans.fit(X_test_scaled)
    #Creating the plot
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15, 10), dpi = 600)
    #styles for title: normal; italic; oblique
    ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c = train_clusters.labels_)
    ax[0].set_title('Plotted Principal Components of TRAIN DATA', style = 'oblique')
    ax[0].legend(f'{int(kmeans.n_clusters)} clusters')
    ax[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c = test_clusters.labels_)
    ax[1].set_title('Plotted Principal Components of TEST DATA', style = 'oblique')
    ax[1].legend(f'{int(kmeans.n_clusters)} clusters')
    #principal components of bank panel has better results than card panel with clearer borders

    #isCredit_num = [1 if x == 'Y' else 0 for x in isCredits]
    #np.corrcoef(np.array(isCredit_num), amounts)

    return X_train, X_train_scaled, X_train_minmax, X_train_pca, X_test, X_test_scaled, X_test_minmax, X_test_pca, y_train, y_test
#%%
@app.route('/pipeline_rfe')
def pipeline_rfe():

    """
    #TEST_RESULTS 4/23/2020 - all unscaled
    #Selected features: ['amount', 'description', 'post_date', 'file_created_date',
    #'optimized_transaction_date', 'panel_file_created_date', 'account_score', 'amount_std_lag3']
    #--
    #TEST_RESULTS 5/6/2020 - all unscaled
    Selected features: ['description', 'post_date', 'file_created_date', 'optimized_transaction_date',
                        'panel_file_created_date', 'account_score', 'amount_std_lag3', 'amount_std_lag7']
    """



    #cols = [c for c in bank_df if bank_df[c].dtype == 'int64' or 'float64']
    #X_train = bank_df[cols].drop(columns = ['primary_merchant_name'], axis = 1)
    #y_train = bank_df['primary_merchant_name']
    #X_test = bank_df[cols].drop(columns = ['primary_merchant_name'], axis = 1)
    #y_test = bank_df['primary_merchant_name']

    #build a logistic regression and use recursive feature elimination to exclude trivial features
    log_reg = LogisticRegression(C = 1.0, max_iter = 2000)
    # create the RFE model and select most striking attributes
    rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
    rfe = rfe.fit(X_train, y_train)
    #selected attributes
    print('Selected features: %s' % list(X_train.columns[rfe.support_]))
    print(rfe.ranking_)
    #following df contains only significant features
    X_train_rfe = X_train[X_train.columns[rfe.support_]]
    X_test_rfe = X_test[X_test.columns[rfe.support_]]
    #log_reg_param = rfe.set_params(C = 0.01, max_iter = 200, tol = 0.001)
    return X_train_rfe, X_test_rfe
#%%
@app.route('/pipeline_rfe_cv')
def pipeline_rfe_cv():


    """
        Application of Recursive Feature Extraction - Cross Validation
        IMPORTANT
        Accuracy: for classification problems
        Mean Squared Error(MSE); Root Mean Squared Error(RSME); R2 Score: for regression
TEST RESULTS
SGDReg
    Completeness Score
    Completeness metric of a cluster labeling given a ground truth.

        A clustering result satisfies completeness if all the data points
        that are members of a given class are elements of the same cluster.

        This metric is independent of the absolute values of the labels:
        a permutation of the class or cluster label values won't change the
        score value in any way.

        This metric is not symmetric: switching ``label_true`` with ``label_pred``
        will return the :func:`homogeneity_score` which will be different in
        general.
    Optimal number of features: 9
    Selected features: ['amount', 'description', 'post_date', 'file_created_date',
                        'optimized_transaction_date', 'panel_file_created_date',
                        'account_score', 'amount_std_lag7', 'amount_std_lag30']
    Max Error -picks all features - BUT HAS GOOD CV SCORE
    Neg Mean Squared Error - picks only one feat
    Homogeneity Score
    Optimal number of features: 9
    Selected features: ['description', 'post_date', 'file_created_date',
                        'optimized_transaction_date', 'panel_file_created_date', 'account_score',
                        'amount_mean_lag3', 'amount_std_lag3', 'amount_std_lag7']
    EVALUATION METRICS DOCUMENTATION
    https://scikit-learn.org/stable/modules/model_evaluation.html
    """

    #Use the Cross-Validation function of the RFE modul
    #accuracy describes the number of correct classifications
    #LOGISTIC REGRESSION
    est_logreg = LogisticRegression(max_iter = 2000)
    #SGD REGRESSOR
    est_sgd = SGDRegressor(loss='squared_loss',
                                penalty='l1',
                                alpha=0.001,
                                l1_ratio=0.15,
                                fit_intercept=True,
                                max_iter=1000,
                                tol=0.001,
                                shuffle=True,
                                verbose=0,
                                epsilon=0.1,
                                random_state=None,
                                learning_rate='constant',
                                eta0=0.01,
                                power_t=0.25,
                                early_stopping=False,
                                validation_fraction=0.1,
                                n_iter_no_change=5,
                                warm_start=False,
                                average=False)
    #SUPPORT VECTOR REGRESSOR
    est_svr = SVR(kernel = 'linear',
                      C = 1.0,
                      epsilon = 0.01)

    #WORKS WITH LOGREG(pick r2), SGDRregressor(r2;rmse)
    rfecv = RFECV(estimator = est_logreg,
                  step = 2,
    #cross_calidation determines if clustering scorers can be used or regression based!
    #needs to be aligned with estimator
                  cv = None,
                  scoring = 'completeness_score')
    rfecv.fit(X_train, y_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    rfecv_num_features = rfecv.n_features_
    print('Selected features: %s' % list(X_train.columns[rfecv.support_]))
    rfecv_features = X_train.columns[rfecv.support_]

    #plot number of features VS. cross-validation scores
    plt.figure(figsize = (10,7))
    plt.suptitle(f"{RFECV.get_params}")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    return rfecv_features, rfecv_num_features
#%%
@app.route('/pipeline_logreg')
def pipeline_logreg():

    """
    1.) split up the data
    2.) fit to the pipeline
    """


    '''
                Setting up a pipeline
    Pipeline 1 - SelectKBest and Logistic Regression (non-neg only)
        PRIMARY_MERCHANT_NAME
    Pipeline 1; 2020-04-29 11:02:06
    {'feature_selection__k': 5, 'reg__max_iter': 800}
    Overall score: 0.3696
    Best accuracy with parameters: 0.34202115158636903
    Pipeline 1; 2020-05-01 09:44:29
    {'feature_selection__k': 8, 'reg__max_iter': 800}
    Overall score: 0.5972
    Best accuracy with parameters: 0.605607476635514
        CITY
    Pipeline 1; 2020-05-04 14:38:23 Full Set
    {'feature_selection__k': 8, 'reg__max_iter': 800}
    Overall score: 0.7953
    Best accuracy with parameters: 0.8155763239875389
    ----
    Pipeline 1; 2020-05-04 17:00:59 Sparse Set
    {'feature_selection__k': 5, 'reg__max_iter': 800}
    Overall score: 0.4706
    Best accuracy with parameters: 0.5158026283963557

    #SelectKBest picks features based on their f-value to find the features that can optimally predict the labels
    #F_CLASSIFIER;FOR CLASSIFICATION TASKS determines features based on the f-values between features & labels;
    #Chi2: for regression tasks; requires non-neg values
    #other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression

    takes unscaled numerical so far and minmax scaled arguments
    #numerical and minmax scaled leads to the same results being picked
    f_classif for classification tasks
    chi2 for regression tasks
    '''

    #Create pipeline with feature selector and regressor
    #replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('reg', LogisticRegression(random_state = 12))])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERS NEED TO HAVE THE SAME LENGTH
    params = {
        'feature_selection__k':[5, 6, 7, 8, 9],
        'reg__max_iter':[800, 1000, 1500, 2000],
        'reg__C':[10, 1, 0.1, 0.01, 0.001],
        }

    #Initialize the grid search object
    grid_search_lr = GridSearchCV(pipe, param_grid = params)

    #best combination of feature selector and the regressor
    #grid_search.best_params_
    #best score
    #grid_search.best_score_

    #Fit it to the data and print the best value combination
    print(f"Pipeline logreg; {dt.today()}")
    print(grid_search_lr.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_lr.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_lr.best_score_}")

    return grid_search_lr
#%%
@app.route('/pipeline_sgd_reg')
def pipeline_sgd_reg():

    '''
    Pipeline 2 - SelectKBest and SGDRegressor -needs non-negative values
    Pipeline 2; 2020-04-29 14:13:46
    {'feature_selection__k': 5, 'reg__alpha': 0.0001, 'reg__max_iter': 800}
    Overall score: -12552683945869548245665121782413383849471150345158656.0000
    Best accuracy with parameters: -1.459592722067248e+50
    '''
    #Create pipeline with feature selector and regressor
    #replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('reg', SGDRegressor(loss='squared_loss', penalty='l1'))
        ])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERS NEED TO HAVE THE SAME LENGTH
    params = {
        'feature_selection__k':[5, 6, 7],
        'reg__alpha':[0.01, 0.001, 0.0001],
        'reg__max_iter':[800, 1000, 1500]
        }

    #Initialize the grid search object
    grid_search_sgd = GridSearchCV(pipe, param_grid = params)

    #Fit it to the data and print the best value combination
    print(f"Pipeline sgd-reg; {dt.today()}")
    print(grid_search_sgd.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_sgd.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_sgd.best_score_}")

    return grid_search_sgd
#%%
@app.route('/pipeline_rfr')
def pipeline_rfr():

    '''
    Pipeline 3 - SelectKBest and Random Forest Regressor
        PRIMARY_MERCHANT_NAME
    ---------
    Pipeline 3; 2020-04-29 11:13:21
    {'feature_selection__k': 7, 'reg__min_samples_split': 8, 'reg__n_estimators': 150}
    Overall score: 0.6965
    Best accuracy with parameters: 0.6820620369181245
    ---
    Pipeline 3; 2020-05-01 10:01:18
    {'feature_selection__k': 7, 'reg__min_samples_split': 4, 'reg__n_estimators': 100}
    Overall score: 0.9319
    Best accuracy with parameters: 0.9181502112642107
        CITY
    Pipeline 3; 2020-05-04 14:50:00 Full Set
    {'feature_selection__k': 7, 'reg__min_samples_split': 4, 'reg__n_estimators': 100}
    Overall score: 0.8422
    Best accuracy with parameters: 0.8558703875627366
    ---
    Pipeline 3; 2020-05-04 17:10:16 Sparse Set
    {'feature_selection__k': 7, 'reg__min_samples_split': 4, 'reg__n_estimators': 150}
    Overall score: 0.7186
    Best accuracy with parameters: 0.75653465869764
    ---
    Pipeline 3; 2020-05-06 10:13:08 with kbest features
    {'feature_selection__k': 5, 'reg__min_samples_split': 8, 'reg__n_estimators': 150}
    Overall score: 0.6255
    Best accuracy with parameters: 0.5813314519498283
    ---
    Pipeline 3; 2020-05-06 16:02:09 Amount_mean_lag7
    {'feature_selection__k': 5, 'reg__min_samples_split': 4, 'reg__n_estimators': 100}
    Overall score: 0.9641
    Best accuracy with parameters: 0.9727385020905415
    '''
    #Create pipeline with feature selector and classifier
    #replace with gradient boosted at this point or regessor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('reg', RandomForestRegressor(n_estimators = 75,
                                      max_depth = len(bank_df.columns)/2,
                                      min_samples_split = 4))
        ])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERS NEED TO HAVE THE SAME LENGTH
    params = {
        'feature_selection__k':[5, 6, 7, 8, 9],
        'reg__n_estimators':[75, 100, 150, 200],
        'reg__min_samples_split':[4, 8, 10, 15],
        }

    #Initialize the grid search object
    grid_search_rfr = GridSearchCV(pipe, param_grid = params)

    #Fit it to the data and print the best value combination
    print(f"Pipeline rfr; {dt.today()}")
    print(grid_search_rfr.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_rfr.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_rfr.best_score_}")

    return grid_search_rfr
#%%
@app.route('/pipeline_svr')
def pipeline_svr():

    '''
    Pipeline 4 - Logistic Regression and Support Vector Kernel -needs non-negative values
    ---------
    Pipeline 4; 2020-05-01 10:06:03
    {'feature_selection__k': 8, 'reg__C': 0.1, 'reg__epsilon': 0.3}
    Overall score: 0.1292
    Best accuracy with parameters: 0.08389477382390549
    --------
        AMOUNT_MEAN_LAG7
    Pipeline 4; 2020-05-06 16:13:22
    {'feature_selection__k': 4, 'reg__C': 1.0, 'reg__epsilon': 0.1}
    Overall score: 0.6325
    Best accuracy with parameters: 0.5934902153570164
    '''
    #Create pipeline with feature selector and regressor
    #replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('reg', SVR(kernel = 'linear'))
        ])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERs NEED TO HAVE THE SAME LENGTH
    #C regularization parameter that is applied to all terms
    #to push down their individual impact and reduce overfitting
    #Epsilon tube around actual values; threshold beyond which regularization is applied
    #the more features picked the more prone the model is to overfitting
    #stricter C and e to counteract
    params = {
        'feature_selection__k':[4, 6, 7, 8, 9],
        'reg__C':[1.0, 0.1, 0.01, 0.001],
        'reg__epsilon':[0.30, 0.25, 0.15, 0.10],
        }

    #Initialize the grid search object
    grid_search_svr = GridSearchCV(pipe, param_grid = params)

    #best combination of feature selector and the regressor
    #grid_search.best_params_
    #best score
    #grid_search.best_score_
    #Fit it to the data and print the best value combination
    print(f"Pipeline svr; {dt.today()}")
    print(grid_search_svr.fit(X_train_minmax, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_svr.score(X_test_minmax, y_test)))
    print(f"Best accuracy with parameters: {grid_search_svr.best_score_}")

    return grid_search_svr
#%%
#BUGGED
# '''
# Pipeline 5 - SelectKBest and Gradient Boosting Classifier
# '''
# #Create pipeline with feature selector and classifier
# pipe = Pipeline([
#     ('feature_selection', SelectKBest(score_func = f_classif)),
#     ('clf', GradientBoostingClassifier(random_state = 42))])

# #Create a parameter grid
# params = {
#     'feature_selection__k':[1, 2, 3, 4, 5, 6, 7],
#     'clf__n_estimators':[15, 25, 50, 75, 120, 200, 350]}

# #Initialize the grid search object
# grid_search = GridSearchCV(pipe, param_grid = params)

# #Fit it to the data and print the best value combination
# print(f"Pipeline 5; {dt.today()}")
# print(grid_search.fit(X_train, y_train).best_params_)
# print("Overall score: %.4f" %(grid_search.score(X_test, y_test)))
# print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
@app.route('/pipeline_knn')
def pipeline_knn():

    '''
    Pipeline 6 - SelectKBest and K Nearest Neighbor
    ----------
    Pipeline 6; 2020-04-27 11:00:27
    {'clf__n_neighbors': 7, 'feature_selection__k': 3}
    Best accuracy with parameters: 0.5928202115158637
    ------
    Pipeline 6; 2020-04-29 10:01:21 WITH SCALED DATA
    {'clf__n_neighbors': 4, 'feature_selection__k': 3}
    Overall score: 0.3696
    Best accuracy with parameters: 0.6156286721504113
    -------
    Pipeline 6; 2020-05-01 10:21:01
    {'clf__n_neighbors': 2, 'feature_selection__k': 4}
    Overall score: 0.9243
    Best accuracy with parameters: 0.9015576323987539
    -------
        CITY
    Pipeline 6; 2020-05-04 14:51:44 Full Set
    {'clf__n_neighbors': 2, 'feature_selection__k': 3}
    Overall score: 0.9028
    Best accuracy with parameters: 0.9071651090342681
    ---
    Pipeline 6; 2020-05-04 17:12:14 Sparse Set
    {'clf__n_neighbors': 3, 'feature_selection__k': 5}
    Overall score: 0.6926
    Best accuracy with parameters: 0.7287349834717407
    ---
        AMOUNT_MEAN_LAG7
    Pipeline 6; 2020-05-06 16:15:12
    {'clf__n_neighbors': 2, 'feature_selection__k': 1}
    Overall score: 0.1157
    Best accuracy with parameters: 0.154583568491494
    '''
    #Create pipeline with feature selector and classifier
    #replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('clf', KNeighborsClassifier())])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERS NEED TO HAVE THE SAME LENGTH
    params = {
        'feature_selection__k':[1, 2, 3, 4, 5, 6, 7],
        'clf__n_neighbors':[2, 3, 4, 5, 6, 7, 8]}

    #Initialize the grid search object
    grid_search_knn = GridSearchCV(pipe, param_grid = params)

    #Fit it to the data and print the best value combination
    print(f"Pipeline knn; {dt.today()}")
    print(grid_search_knn.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_knn.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_knn.best_score_}")
    return grid_search_knn
#%%
@app.route('/pipeline_svc')
def pipeline_svc():

    '''
    Pipeline 7 - SelectKBest and Support Vector Classifier
    ##########
    Pipeline 7; 2020-04-28 10:22:10
    {'clf__C': 100, 'clf__gamma': 0.1, 'feature_selection__k': 5}
    Best accuracy with parameters: 0.6742596944770858
    ---
    Pipeline 7; 2020-04-29 10:06:28 SCALED DATA
    {'clf__C': 0.01, 'clf__gamma': 0.1, 'feature_selection__k': 4}
    Overall score: 0.3696
    Best accuracy with parameters: 0.34202115158636903
    ---
    Pipeline 7; 2020-04-29 10:11:02 UNSCALED DATA
    {'clf__C': 10, 'clf__gamma': 0.01, 'feature_selection__k': 5}
    Overall score: 0.5266
    Best accuracy with parameters: 0.5592068155111634
    ---
    Pipeline 7; 2020-04-30 11:38:13
    {'clf__C': 1, 'clf__gamma': 0.01, 'feature_selection__k': 4}
    Overall score: 0.5408
    Best accuracy with parameters: 0.5335967104732726
    ---
    Pipeline 7; 2020-05-01 10:29:08
    {'clf__C': 100, 'clf__gamma': 0.01, 'feature_selection__k': 4}
    Overall score: 0.9346
    Best accuracy with parameters: 0.9102803738317757
    ---
    Pipeline 7; 2020-05-04 10:52:47
    {'clf__C': 10, 'clf__gamma': 0.1, 'feature_selection__k': 4}
    Overall score: 0.9121
    Best accuracy with parameters: 0.9171339563862928
    ---
        CITY
    Pipeline 7; 2020-05-04 14:58:15 Full Set
    {'clf__C': 10, 'clf__gamma': 0.01, 'feature_selection__k': 5}
    Overall score: 0.8841
    Best accuracy with parameters: 0.8797507788161993
    ---
    Pipeline 7; 2020-05-04 17:14:48 Sparse Set
    {'clf__C': 10, 'clf__gamma': 0.1, 'feature_selection__k': 5}
    Overall score: 0.7533
    Best accuracy with parameters: 0.7908651132790454
    ---
        AMOUNT-MEAN-LAG7
    Pipeline 7; 2020-05-06 16:17:40
    {'clf__C': 10, 'clf__gamma': 0.001, 'feature_selection__k': 4}
    Overall score: 0.1044
    Best accuracy with parameters: 0.16726598403612028
    '''
    #Create pipeline with feature selector and classifier
    #replace with classifier or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('clf', SVC())])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERS NEED TO HAVE THE SAME LENGTH
    #Parameter explanation
    #   C: penalty parameter
    #   gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    #
    params = {
        'feature_selection__k':[4, 5, 6, 7, 8, 9],
        'clf__C':[0.01, 0.1, 1, 10],
        'clf__gamma':[0.1, 0.01, 0.001]}

    #Initialize the grid search object
    grid_search_svc = GridSearchCV(pipe, param_grid = params)

    #Fit it to the data and print the best value combination
    print(f"Pipeline 7; {dt.today()}")
    print(grid_search_svc.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_svc.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_svc.best_score_}")
    return grid_search_svc
#%%
#generate a dataframe for pipeline values
def score_df(gs_object):
    gs_df = pd.DataFrame(data = {'Given Parameters':gs_object.param_grid,
                                'Best Parameters':gs_object.best_params_,
                                'Mean Score':gs_object.score(X_test, y_test),
                                'Mean Score(scaled)':gs_object.score(X_test_scaled, y_test),
                                'Highest Prediction Score':gs_object.best_score_
                                })
    print(gs_df)
#%%
'''
Catching the predictions and converting them back to merchants
Should the prediction turn out to be wrong ask for input by the user
Label needs to be primary_merchant_name
'''
def merch_pred():
    merch_list = []
    for merchant, value in embedding_map_merchants.items():
        for prediction in grid_search.predict(X_test):
            if prediction == value:
                #print(f"Transaction at {merchant}")
                merch_list.append(merchant)
            # else:
            #     print("This merchant could not be recognized by us.\nCan you tell us where you are shopping right now? :)")
            #     merch_list.append("Wrong prediction")
    return merch_list
#%%
#BUGGED
'''
        Catching the predictions and converting them back to lagging amounts
Should the prediction turn out to be wrong ask for input by the user
'''
def amount_pred():
    #append pred to a list and ten compare current value with previous value
    #weekly_mean = []
    budget_dict = {}
    #weekly_exp_list = enumerate(grid_search.predict(X_test))

    for index, i in enumerate(grid_search.predict(X_test)):
        print(index, i)
        #weekly_mean.append(i)

        if i > [i-1]:
            print("Your weekly average expenses increased; more money will be put into savings")
            msg_1 = "Your weekly average expenses increased"

            budget_dict[key].append(msg_1)
        elif i < [i-1]:
            print("Your weekly average expenses decreased. Keep up the good budgeting")
            msg_2 = "Your weekly average expenses decreased"

            budget_dict[key].append(msg_2)
        else:
            print("Your expenses are stable")
            msg_3 = "Your expenses are stable"

            budget_dict[key].append(msg_3)
#%%
@app.route('/pipeline_trans_reg')
def pipeline_trans_reg():

    '''
            Application of Transformed Linear Regression

    #n_quantiles needs to be smaller than the number of samples (standard is 1000)

    PRIMARY_MERCHANT_NAME
    #accuracy negative; model totally off
    ---
    AMOUNT_MEAN_LAG7
    q-t R2-score: 0.896
    unprocessed R2-score: 0.926
    '''
    transformer = QuantileTransformer(n_quantiles=750, output_distribution='normal')
    regressor = LinearRegression()
    regr = TransformedTargetRegressor(regressor=regressor,
                                       transformer=transformer)

    regr.fit(X_train, y_train)

    TransformedTargetRegressor(...)
    print('q-t R2-score: {0:.3f}'.format(regr.score(X_test, y_test)))

    raw_target_regr = LinearRegression().fit(X_train, y_train)
    print('unprocessed R2-score: {0:.3f}'.format(raw_target_regr.score(X_test, y_test)))
    return regr, raw_target_regr
#%%
#Overfitting - model will not be used
'''
    Random Forest Classifier
'''
# RFC = RandomForestClassifier(n_estimators = 20, max_depth = len(bank_df.columns) /2, random_state = 7)
# RFC.fit(X_train, y_train)
# y_pred = RFC.predict(X_test)
# RFC_probability = RFC.predict_proba(X_test)
# print(f"TESTINFO Rnd F Cl: [{dt.today()}]--[Parameters: n_estimators:{RFC.n_estimators},\
#       max_depth:{RFC.max_depth}, random state:{RFC.random_state}]--Training set accuracy:\
#       {RFC.score(X_train, y_train)}; Test set accuracy: {RFC.score(X_test, y_test)};\
#           Test set validation: {RFC.score(X_test, y_pred)}")
#%%
#Overfitting - model will not be used
'''
    K Nearest Neighbor
'''
# KNN = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform')
# KNN.fit(X_train, y_train)
# y_pred = KNN.predict(X_test)
# print(f"TESTINFO KNN: [{dt.today()}]--[Parameters: n_neighbors:{KNN.n_neighbors},\
#       weights:{KNN.weights}]--Training set accuracy: {KNN.score(X_train, y_train)};\
#       Test set accuracy: {KNN.score(X_test, y_test)}; Test set validation: {KNN.score(X_test, y_pred)}")
#%%
#Overfitting - model will not be used
'''
    Random Forest Regressor

Use the random forest regressor algorithm to predict labels; DO NOT USE SCALED VARIABLES HERE
The number of splits for each tree level is equal to half the number of columns; that way overfitting is dampened and test remains fast
Test 4/22/2020: val_accuracy: 1.0 -> overfitted
'''
# RFR = RandomForestRegressor(n_estimators = 75, max_depth = len(bank_df.columns)/2, min_samples_split = 4)
# RFR.fit(X_train, y_train)
# y_pred = RFR.predict(X_test)
# print(f"TESTINFO Rnd F Reg: [{dt.today()}]--[Parameters: n_estimators:{RFR.n_estimators},\
#       max_depth:{RFR.max_depth}, min_samples_split:{RFR.min_samples_split}]--Training set accuracy:\
#       {RFR.score(X_train, y_train)}; Test set accuracy: {RFR.score(X_test, y_test)}; Test set validation:\
#           {RFR.score(X_test, y_pred)}")
#%%
#model is overfitting extremely
# def mlp_raw_pred():

#     '''
#                     APPLICATION OF SKLEARN NEURAL NETWORK

#     Test; [1000l;alpha=0.001] [2020-04-22 00:00]; Training set accuracy: 0.002171552660152009;\
#                                                                         Test set accuracy: 0.0
#     Test; [1500l; alpha=0.0001] [2020-04-22 14:16:43]; Training set accuracy: 0.3517915309446254;
#                                                        Test set accuracy: 0.3468354430379747
#     Test; [2020-04-29 14:32:25]--[Parameters: hidden layers:1500, alpha:0.0001]--\
#                                                 Training set accuracy: 0.35868187579214195;\
#                                                 Test set accuracy: 0.3377609108159393;\
#                                                 Test set Validation: 1.0
#     Test; [2020-05-01 10:31:28.213131]--[Parameters: hidden layers:1500, alpha:0.0001]--\
#                                             Training set accuracy: 0.33084112149532713;\
#                                             Test set accuracy: 0.3317757009345794;\
#                                             Test set Validation: 1.0
#     '''
#     #adam: all-round solver for data
#     #hidden_layer_sizes: no. of nodes/no. of hidden weights used to obtain final weights;
#     #match with input features
#     #alpha: regularization parameter that shrinks weights toward 0 (the greater the stricter)
#     MLP = MLPClassifier(hidden_layer_sizes = 1500, solver='sgd', learning_rate= 'adaptive', alpha=0.0001)
#     MLP.fit(X_train, y_train)
#     y_val = MLP.predict(X_test)
#     #y_val.reshape(-1, 1)
#     print(f"TESTINFO MLP: [{dt.today()}]--[Parameters: hidden layers:{MLP.hidden_layer_sizes}, alpha:{MLP.alpha}]--\
#           Training set accuracy: {MLP.score(X_train, y_train)}; Test set accuracy: {MLP.score(X_test, y_test)};\
#           Test set Validation: {MLP.score(X_test, y_val)}")
#     return 'Multi-layered perceptron object ready'
#%%
@app.route('/pipeline_mlp')
def pipeline_mlp():

    '''
    Pipeline 8 - SelectKBest and Multi-Layer Perceptron
    ##########
    Pipeline 7; 2020-05-06 10:20:51 CITY (sgd + adaptive learning rate)
    {'clf__alpha': 0.0001, 'clf__max_iter': 2000, 'feature_selection__k': 5}
    Overall score: 0.2808
    Best accuracy with parameters: 0.26102555833266144
    '''
    #Create pipeline with feature selector and classifier
    #replace with classifier or regressor
    #learning_rate = 'adaptive'; when solver='sgd'
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('clf', MLPClassifier(activation='relu',
                              solver='lbfgs',
                              learning_rate='constant'))])

    #Create a parameter grid
    #Parameter explanation
    #   C: penalty parameter
    #   gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    #
    params = {
        'feature_selection__k':[4, 5, 6, 7],
        'clf__max_iter':[1500, 2000],
        'clf__alpha':[0.0001, 0.001]}

    #Initialize the grid search object
    grid_search_mlp = GridSearchCV(pipe, param_grid = params)

    #Fit it to the data and print the best value combination
    print(f"Pipeline 7; {dt.today()}")
    print(grid_search_mlp.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_mlp.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_mlp.best_score_}")
    return grid_search_mlp
#%%
@app.route('/pipeline_mlpreg')
def pipeline_mlpreg():

    '''
    Pipeline 8 - SelectKBest and Multi-Layer Perceptron Regressor
    ##########
    Pipeline 8; 2020-05-20 11:50:43
    {'clf__alpha': 0.001, 'clf__max_iter': 1200, 'feature_selection__k': 5}
    Overall score: 0.9632
    Best accuracy with parameters: 0.9623355019137264
    '''
    #Create pipeline with feature selector and classifier
    #replace with classifier or regressor
    #learning_rate = 'adaptive'; when solver='sgd'
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('clf', MLPRegressor(activation='relu',
                              solver='lbfgs',
                              learning_rate='constant'))])

    #Create a parameter grid
    #Parameter explanation
    #   C: penalty parameter
    #   gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    #
    params = {
        'feature_selection__k':[4, 5, 6, 7],
        'clf__max_iter':[800, 1200, 1500],
        'clf__alpha':[0.0001, 0.001, 0.01]}

    #Initialize the grid search object
    grid_search_mlpreg = GridSearchCV(pipe, param_grid = params)

    #Fit it to the data and print the best value combination
    print(f"Pipeline 8; {dt.today()}")
    print(grid_search_mlpreg.fit(X_train_minmax, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_mlpreg.score(X_test_minmax, y_test)))
    print(f"Best accuracy with parameters: {grid_search_mlpreg.best_score_}")

    return grid_search_mlpreg
#%%
#flask connection in respective pipeline folder
def store_pickle(model):

    """
    Usage of a Pickle Model -Storage of a trained Model
    """

    model_file = "gridsearch_model.sav"
    with open(model_file, mode='wb') as m_f:
        pickle.dump(model, m_f)
    print(f"Model saved in: {os.getcwd()}")
    return model_file
#%%
#flask connection in respective pipeline folder
def open_pickle(model_file):

    """
        Usage of a Pickle Model -Loading of a Pickle File

    model file can be opened either with FILE NAME
    open_pickle(model_file="gridsearch_model.sav")
    INTERNAL PARAMETER
    open_pickle(model_file=model_file)
    """

    with open(model_file, mode='rb') as m_f:
        grid_search = pickle.load(m_f)
        result = grid_search.score(X_test, y_test)
        print("Employed Estimator:", grid_search.get_params)
        print("--------------------")
        print("BEST PARAMETER COMBINATION:", grid_search.best_params_)
        print("Training Accuracy Result: %.4f" %(result))
        return 'grid_search parameters loaded'
#%%
#y_val = y_pred as the split is still unfisnished
print('R2 score = ', r2_score(y_val, y_test), '/ 1.0')
print('MSE score = ', mean_squared_error(y_val, y_test), '/ 0.0')
#%%
'''
                        APPLICATION OF KERAS
'''
#features: X
#target: Y

features = np.array(X_train)
targets = np.array(y_train)
features_validation = np.array(X_test)
targets_validation = np.array(y_test)

print(features[:10])
print(targets[:10])
#%%
#building the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(1))

#compiling the model
model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) #mse: mean_square_error
model.summary()
#%%
#training of the model
epochs_total = 1000
epochs_step = 250
epochs_ratio = int(epochs_total / epochs_step)
hist = np.array([])
#######
#building the epochs
for i in range(epochs_ratio):
    history = model.fit(features, targets, epochs=epochs_step, batch_size=100, verbose=0)

    #evaluating the model on the training and testing set
    print("Step : " , i * epochs_step, "/", epochs_total)
    score = model.evaluate(features, targets)
    print("Training MSE:", score[1])
    score = model.evaluate(features_validation, targets_validation)
    print("Validation MSE:", score[1], "\n")
    #hist = np.concatenate((hist, np.array(history.history['mean_squared_error'])), axis = 0)
#%%

#plot metrics
plt.plot(hist)
plt.show()

predictions = model.predict(features_validation, verbose=0)
print('R2 score = ', r2_score(y_test, predictions), '/ 1.0')
print('MSE score = ', mean_squared_error(targets_validation, predictions), '/ 0.0')
#######
plt.plot(features_validation[0:50], '+', color ='blue', alpha=0.7)
plt.plot(predictions[0:50], 'ro', color ='red', alpha=0.5)
plt.show()
#%%
'''
                    APPLICATION OF TENSORFLOW
'''
#custom train test split
train, test = train_test_split(bank_df, test_size = 0.2)
train, val = train_test_split(train, test_size = 0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
#%%
#create tuples for lists to organize the columns conversion
##Tuple (or list) for the bank list to refer to the length

##STEP 1
'''
#setting up various columns to be features and encode them to be ready for a NN
layer
'''
#feature columns to use in the layers
feature_columns_container = []

#numeric column needed in the model
#wrap all non-numerical columns with indicator col or embedding col
#the argument default_value governs out of vocabulary values and how to replace them

for header in bank_df.columns:
    feature_columns_container.append(feature_column.numeric_column(header))

#indicator columns
city = feature_column.numeric_column("city")
city_ind = feature_column.indicator_column(city)
feature_columns_container.append(city_ind)

#BUCKETIZED COLUMN
#categorical column with vocabulary list
type_col = feature_column.categorical_column_with_vocabulary_list(
        'type', ['CorePro Deposit',
                 'CorePro Withdrawal',
                 'Internal CorePro Transfer',
                 'Interest Paid',
                 'CorePro Recurring Withdrawal',
                 'Manual Adjustment',
                 'Interest Adjustment'])
type_pos = feature_column.indicator_column(type_col)
type_pos_2 = feature_column.embedding_column(type_col, dimension = 8)
feature_columns_container.append(type_pos)
feature_columns_container.append(type_pos_2)

#idea: words or fragments are a bucket and can be used to recognize recurring bills
friendly_desc = feature_column.categorical_column_with_hash_bucket(
        'friendlyDescription', hash_bucket_size = 2500)
fr_desc_pos = feature_column.embedding_column(friendly_desc, dimension = 250)
feature_columns_container.append(fr_desc_pos)

#created_date = feature_column.categorical_column_with_hash_bucket(
        'createdDate', hash_bucket_size = 365)

#set the indicator column
cr_d_pos = feature_column.indicator_column(created_date)
feature_columns_container.append(cr_d_pos)


#fee = feature_column.categorical_column_with_vocabulary_list('feeCode',
                                                             ['RGD',
                                                              'RTN',
                                                              'NSF'])
#set the embedding column
fee_pos = feature_column.embedding_column(fee, dimension = 3)
feature_columns_container.append(fee_pos)

FICO 700 is the initial score and also the average in the US
The CS_FICO_num column is in this version converted to a bucketized column
instead of passing it to the feature_column_container
#columns remains bucketized without wrapping to embedded or indicator
fico_num = feature_column.bucketized_column(feature_column.numeric_column('CS_FICO_num'),
                                                boundaries = [300,
                                                              580,
                                                              670,
                                                              700,
                                                              740,
                                                              800,
                                                              850])
feature_columns_container.append(fico_num)


institutions = feature_column.categorical_column_with_vocabulary_list(
        'institutionName', [
            'Bank of America', 'Toronto Dominion Bank', 'Citizens Bank', 'Webster Bank',
            'CHASE Bank', 'Citigroup', 'Capital One', 'HSBC Bank USA',
            'State Street Corporation', 'MUFG Union Bank', 'Wells Fargo & Co.', 'Barclays',
            'New York Community Bank', 'CIT Group', 'Santander Bank',
            'Royal Bank of Scotland', 'First Rand Bank', 'Budapest Bank'
            ])
institutions_pos = feature_column.indicator_column(institutions)
feature_columns_container.append(institutions_pos)

crossed_feat = feature_column.crossed_column([age, fico_num], hash_bucket_size = 1000)
crossed_feat = feature_column.indicator_column(crossed_feat)
feature_columns_container.append(crossed_feat)

###########EXAMPLES#######
#numeric column
#age = feature_column.numeric_column("age")

#categorical column with vocabulary list
#thal = feature_column.categorical_column_with_vocabulary_list(
#      'thal', ['fixed', 'normal', 'reversible'])

#bucketized column
#age_buckets = feature_column.bucketized_column(
#   age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

#embedding column
#feature_column.embedding_column(thal, dimension=8)
#feature_columns_container.append(age_buckets)

#hashed feature column
#feature_column.categorical_column_with_hash_bucket(
#      'thal', hash_bucket_size=1000)
#feature_columns_container.append(age_buckets)

#crossed feature column
#feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
#feature_columns_container.append(age_buckets)

#indicator column (like bucketized but with one vital string that is marked a "1")
#also used as a wrapper for categorical columns to ensure wokring feature_layers
##########################
#%%
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle = True, batch_size = len(dataframe.index/3)):
    #dataframe = dataframe.copy()
    label = dataframe.pop('city')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), label))
    if shuffle:
        #buffer size is read into memory!
        ds = ds.shuffle(buffer_size = 512)
        ds = ds.batch(batch_size)

    return ds
#%%
##STEP 2
#create layers
feature_layer = tf.keras.layers.DenseFeatures(feature_columns_container)
#print(feature_layer)
#%%
#STEP 3
batch_size = 250
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle = True, batch_size = batch_size)
test_ds = df_to_dataset(test, shuffle = True, batch_size = batch_size)
#%%
#STEP 4
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(units = 256, activation = 'relu'),
  layers.Dense(units = 256, activation = 'relu'),
  layers.Dense(units = 256, activation = 'relu'),
  layers.Dense(units = 1, activation = 'sigmoid')
])
#%%
##STEP 5
model.compile(optimizer = 'Adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(train_ds,
          validation_data = val_ds,
          epochs=2)
#%%
##STEP 6
#Check accuracy
loss, accuracy = model.evaluate(test_ds)
print("Accuracy:", accuracy)

##STEP 7
# Infer labels on a batch
predictions = model.predict(test_ds)
#%%
'''
Application of an Unsupervised Learning Algorithms
'''
#local outlier frequency
#Contamination to match outlier frequency in ground_truth
preds = LocalOutlierFactor(contamination=0.2).fit_predict(X_train)
#Print the confusion matrix
print(confusion_matrix(y_train, preds))

#anti-fraud system + labeling
#pick parameters to spot outliers in

#while loop to stop as soon s first income is hit and add upp income/expense

#pass this to flask and app to inject this initial balance
