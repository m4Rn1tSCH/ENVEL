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

#load needed packages
import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np
from datetime import datetime as dt
import os
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_selection import SelectKBest , chi2, f_classif, RFE, RFECV
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.cluster import KMeans

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline


import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

#imported custom function
#generates a CSV for daily/weekly/monthly account throughput; expenses and income
from Python_spending_report_csv_function import spending_report
#contains the connection script
from Python_SQL_connection import execute_read_query, create_connection, close_connection
#contains all credentials
import PostgreSQL_credentials as acc
#loads flask into the environment variables
#from flask_auto_setup import activate_flask
#csv export with optional append-mode
from Python_CSV_export_function import csv_export
#%%
#set up flask first as environment variable and start with the command console
#activate_flask()
    #CONNECTION TO FLASK/SQL
#app = Flask(__name__)

##put address here
#function can be bound to the script by adding a new URL
#e.g. route('/start') would then start the entire function that follows
#same can be split up
#@app.route('/encode')

def df_encoder(rng = 4):
    '''

    Parameters
    ----------
    rng : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    '''
    #%%
    connection = create_connection(db_name = acc.YDB_name,
                                   db_user = acc.YDB_user,
                                   db_password = acc.YDB_password,
                                   db_host = acc.YDB_host,
                                   db_port = acc.YDB_port)
    #%%
    #establish connection to get user IDs
    filter_query = f"SELECT unique_mem_id, state, city, zip_code, income_class, file_created_date FROM user_demographic WHERE state = 'MA'"
    transaction_query = execute_read_query(connection, filter_query)
    query_df = pd.DataFrame(transaction_query,
                            columns = ['unique_mem_id', 'state', 'city', 'zip_code', 'income_class', 'file_created_date'])
    #%%
    #dateframe to gather MA bank data from one randomly chosen user
    #std random_state is 2
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
    #%%
    '''
    Plotting of various relations
    The Counter object keeps track of permutations in a dictionary which can then be read and
    used as labels
    '''
    #Pie chart States - works
    state_ct = Counter(list(bank_df['state']))
    #asterisk look up, what is that?
    labels, values = zip(*state_ct.items())
    #Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig1, ax = plt.subplots(figsize = (18, 12))
    ax.pie(values, labels = labels, autopct = '%1.1f%%',
            shadow = True, startangle = 90)
    #Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    #ax.title('Transaction locations of user {bank_df[unique_mem_id][0]}')
    ax.legend(loc = 'center right')
    plt.show()

    #Pie chart transaction type -works
    trans_ct = Counter(list(bank_df['transaction_category_name']))
    #asterisk look up, what is that?
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

#BUGGED
    #Boxplot template
    # cat_var = ["unique_mem_id", "primary_merchant_name"]
    # quant_var = ["amount"]
    # for c_var in cat_var:
    #     for q_var in quant_var:
    #         bank_df.boxplot(column=q_var, by=c_var)
    #         plt.xticks([])
    #%%
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
    #%%
    '''
    After successfully loading the data, columns that are of no importance have been removed and missing values replaced
    Then the dataframe is ready to be encoded to get rid of all non-numerical data
    '''
    # print(bank_df[bank_df['city'].isnull()])
    # #Then for remove all not numeric values use to_numeric with parameetr errors='coerce' - it replace non numeric to NaNs:
    # bank_df['x'] = pd.to_numeric(bank_df['x'], errors='coerce')
    # #And for remove all rows with NaNs in column x use dropna:
    # bank_df = bank_df.dropna(subset=['x'])
    # #Last convert values to ints:
    # bank_df['x'] = bank_df['x'].astype(int)
        #prepare numeric and string columns

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
    #%%
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
    #UNKNOWN_TOKEN = '<unknown>'
    states = bank_df['state'].unique().astype('str').tolist()
    states.append(UNKNOWN_TOKEN)
    le_3 = LabelEncoder()
    le_3.fit_transform(states)
    embedding_map_states = dict(zip(le_3.classes_, le_3.transform(le_3.classes_)))

    #APPLICATION TO OUR DATASET
    bank_df['state'] = bank_df['state'].apply(lambda x: x if x in embedding_map_states else UNKNOWN_TOKEN)
    bank_df['state'] = bank_df['state'].map(lambda x: le_3.transform([x])[0] if type(x)==str else x)

    #encoding descriptions
    #UNKNOWN_TOKEN = '<unknown>'
    desc = bank_df['description'].unique().astype('str').tolist()
    desc.append(UNKNOWN_TOKEN)
    le_4 = LabelEncoder()
    le_4.fit_transform(desc)
    embedding_map_desc = dict(zip(le_4.classes_, le_4.transform(le_4.classes_)))

    #APPLICATION TO OUR DATASET
    bank_df['description'] = bank_df['description'].apply(lambda x: x if x in embedding_map_desc else UNKNOWN_TOKEN)
    bank_df['description'] = bank_df['description'].map(lambda x: le_4.transform([x])[0] if type(x)==str else x)

    #encoding descriptions
    #UNKNOWN_TOKEN = '<unknown>'
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
    #UNKNOWN_TOKEN = '<unknown>'
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
    #%%
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

    #FEATURE ENGINEERING II
    #typical engineered features based on lagging metrics
    #mean + stdev of past 3d/7d/30d/ + rolling volume
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
    #%%
    #the first two rows of lagging values have NaNs which need to be dropped
    #drop the first and second row since the indicators refer to previous non-existant days
    bank_df = bank_df.drop([0, 1])
    bank_df.reset_index(drop = True, inplace = True)
    #csv_export(df=bank_df, file_name='encoded_bank_dataframe')
    return 'dataframe encoding complete; CSVs are located in the working directory(inactivated for testing)'
#%%
###################SPLITTING UP THE DATA###########################
#drop target variable in feature df
#all remaining columns will be the features
bank_df = bank_df.drop(['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id'], axis = 1)
model_features = bank_df.drop(['primary_merchant_name'], axis = 1)
model_label = bank_df['primary_merchant_name']

X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                    model_label,
                                                    shuffle = True,
                                                    test_size = 0.3)

#create a validation set from the training set
print(f"Shape of the split training data set X_train:{X_train.shape}")
print(f"Shape of the split training data set X_test: {X_test.shape}")
print(f"Shape of the split training data set y_train: {y_train.shape}")
print(f"Shape of the split training data set y_test: {y_test.shape}")
#%%
#STD SCALING - does not work yet
#fit the scaler to the training data first
#standard scaler works only with maximum 2 dimensions
scaler = StandardScaler(copy = True, with_mean = True, with_std = True).fit(X_train)
X_train_scaled = scaler.transform(X_train)

#transform test data with the object learned from the training data
X_test_scaled = scaler.transform(X_test)
scaler_mean = scaler.mean_
stadard_scale = scaler.scale_
#%%
#MINMAX SCALING - works with Select K Best
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)

X_test_minmax = min_max_scaler.transform(X_test)
minmax_scale = min_max_scaler.scale_
min_max_minimum = min_max_scaler.min_
#%%
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
#%%
'''
            PLotting of PCA/ Cluster Pairs

'''
    #Kmeans clusters to categorize groups WITH SCALED DATA
    #determine number of groups needed or desired for
    kmeans = KMeans(n_clusters = 5, random_state = 10)
    train_clusters = kmeans.fit(X_train_scaled)

    kmeans = KMeans(n_clusters = 5, random_state = 10)
    test_clusters = kmeans.fit(X_test_scaled)
    #%%
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15, 10), dpi = 600)
    #styles for title: normal; italic; oblique
    ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c = train_clusters.labels_)
    ax[0].set_title('Plotted Principal Components of TRAIN DATA', style = 'oblique')
    ax[0].legend(train_clusters.labels_)
    ax[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c = test_clusters.labels_)
    ax[1].set_title('Plotted Principal Components of TEST DATA', style = 'oblique')
    ax[1].legend(test_clusters.labels_)
    #principal components of bank panel has better results than card panel with clearer borders
#%%
#f_classif for regression
#chi-sqr for classification but requires non-neg values
####syntax of reshape(n_samples, n_features)
#### value of -1 allows for adaptation to shape needed
##y_train_rs = np.array(y_train).reshape(-1, 1)
##X_train_scl_rs = np.array(X_train_scaled).reshape(-1, 1)
##X_test_scl_rs = np.array(X_test_scaled).reshape(-1, 1)

##X_train_minmax_rs = X_train_minmax.reshape(-1, 1)
##X_test_minmax_rs = X_test_minmax.reshape(-1, 1)
#%%
#fed variables cannot have missing values
'''
takes unscaled numerical so far and minmax scaled arguments
#numerical and minmax scaled leads to the same results being picked
f_classif for classification tasks
chi2 for regression tasks
'''
k_best = SelectKBest(score_func = f_classif, k = 10)
k_best.fit(X_train_scaled, y_train)
k_best.get_params()

#isCredit_num = [1 if x == 'Y' else 0 for x in isCredits]
#np.corrcoef(np.array(isCredit_num), amounts)
#%%
#WORKS WITH UNSCALED DATA
#pick feature columns to predict the label
#TEST_RESULTS 4/23/2020 - all unscaled
#Selected features: ['amount', 'description', 'post_date', 'file_created_date',
#'optimized_transaction_date', 'panel_file_created_date', 'account_score', 'amount_std_lag3']

#cols = [c for c in bank_df if bank_df[c].dtype == 'int64' or 'float64']
#X_train = bank_df[cols].drop(columns = ['primary_merchant_name'], axis = 1)
#y_train = bank_df['primary_merchant_name']
#X_test = bank_df[cols].drop(columns = ['primary_merchant_name'], axis = 1)
#y_test = bank_df['primary_merchant_name']

#build a logistic regression and use recursive feature elimination to exclude trivial features
log_reg = LogisticRegression(C = 1.0, max_iter = 1500)
# create the RFE model and select most striking attributes
rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
rfe = rfe.fit(X_train, y_train)
#selected attributes
print('Selected features: %s' % list(X_train.columns[rfe.support_]))
print(rfe.ranking_)
#following df contains only significant features
X_train_kbest = X_train[X_train.columns[rfe.support_]]
X_test_kbest = X_test[X_test.columns[rfe.support_]]
#log_reg_param = rfe.set_params(C = 0.01, max_iter = 200, tol = 0.001)
#%%
'''
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
    Selected features: ['amount', 'description', 'post_date', 'file_created_date', 'optimized_transaction_date', 'panel_file_created_date', 'account_score', 'amount_std_lag7', 'amount_std_lag30']
    Max Error -picks all features - BUT HAS GOOD CV SCORE
    Neg Mean Squared Error - picks only one feat
    Homogeneity Score
    Optimal number of features: 9
    Selected features: ['description', 'post_date', 'file_created_date', 'optimized_transaction_date', 'panel_file_created_date', 'account_score', 'amount_mean_lag3', 'amount_std_lag3', 'amount_std_lag7']
EVALUATION METRICS DOCUMENTATION
https://scikit-learn.org/stable/modules/model_evaluation.html
'''
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
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

#plot number of features VS. cross-validation scores
plt.figure(figsize = (10,7))
plt.suptitle(f"{RFECV.get_params}")
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
#%%
'''
            Setting up a pipeline
Pipeline 1 - SelectKBest and Logistic Regression
'''
#SelectKBest picks features based on their f-value to find the features that can optimally predict the labels
#F_CLASSIFIER;FOR CLASSIFICATION TASKS determines features based on the f-values between features & labels;
#Chi2: for regression tasks; requires non-neg values
#other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression

#Create pipeline with feature selector and regressor
#replace with gradient boosted at this point or regressor
pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func = chi2)),
    ('reg', LogisticRegression(C = 1.0, random_state = 42))])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERS NEED TO HAVE THE SAME LENGTH
params = {
    'feature_selection__k':[5, 6, 7],
    'reg__max_iter':[800, 1000, 1500]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#best combination of feature selector and the regressor
#grid_search.best_params_
#best score
#grid_search.best_score_

#Fit it to the data and print the best value combination
print(f"Pipeline 1; {dt.today()}")
print(grid_search.fit(X_train, y_train).best_params_)
print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
'''
Pipeline 2 - SelectKBest and SGDRegressor
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
    'reg__alpha':[0.01, 0.001, 0.0001, 0.000001],
    'reg__max_iter':[800, 1000, 1500, 2500]
    }

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(f"Pipeline 2; {dt.today()}")
print(grid_search.fit(X_train, y_train).best_params_)
print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
#BUGGED
'''
Pipeline 3 - Logistic Regression and Random Forest Regressor
'''
#Create pipeline with feature selector and classifier
#replace with gradient boosted at this point or regessor
pipe = Pipeline([
    ('feature_selection', RFE(estimator = LogisticRegression(C = 1.0, max_iter = 1500),
                              step = 1)),
    ('reg', RandomForestRegressor(n_estimators = 75, max_depth = len(bank_df.columns)/2, min_samples_split = 4))
    ])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERS NEED TO HAVE THE SAME LENGTH
params = {
    'feature_selection__n_features_to_select':[6, 7, 8, 9],
    'reg__n_estimators':[75, 100, 150, 200],
    'reg__min_samples_split':[4, 8, 10, 15],
    }

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(f"Pipeline 3; {dt.today()}")
print(grid_search.fit(X_train, y_train).best_params_)
print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
#BUGGED
'''
Pipeline 4 - Logistic Regression and Support Vector Kernel
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
grid_search = GridSearchCV(pipe, param_grid = params)

#best combination of feature selector and the regressor
#grid_search.best_params_
#best score
#grid_search.best_score_
#Fit it to the data and print the best value combination
print(f"Pipeline 4; {dt.today()}")
print(grid_search.fit(X_train, y_train).best_params_)
print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
'''
Pipeline 5 - SelectKBest and Gradient Boosting Classifier
'''
#Create pipeline with feature selector and classifier
#replace with gradient boosted at this point or regressor
pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func = f_classif)),
    ('clf', GradientBoostingClassifier(random_state = 42))])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERS NEED TO HAVE THE SAME LENGTH
params = {
    'feature_selection__k':[1, 2, 3, 4, 5, 6, 7],
    'clf__n_estimators':[15, 25, 50, 75, 120, 200, 350]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(f"Pipeline 5; {dt.today()}")
print(grid_search.fit(X_train, y_train).best_params_)
print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
'''
Pipeline 6 - SelectKBest and K Nearest Neighbor
##########
Pipeline 6; 2020-04-27 11:00:27
{'clf__n_neighbors': 7, 'feature_selection__k': 3}
Best accuracy with parameters: 0.5928202115158637
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
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(f"Pipeline 6; {dt.today()}")
print(grid_search.fit(X_train, y_train).best_params_)
print(f"Best accuracy with parameters: {grid_search.best_score_}")
#%%
#accuracy negative; model toally off
transformer = QuantileTransformer(output_distribution='normal')
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor,
                                   transformer=transformer)

regr.fit(X_train, y_train)

TransformedTargetRegressor(...)
print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))


raw_target_regr = LinearRegression().fit(X_train, y_train)
print('R2 score: {0:.2f}'.format(raw_target_regr.score(X_test, y_test)))
#%%
'''
Random Forest Classifier
'''
RFC = RandomForestClassifier(n_estimators = 20, max_depth = len(bank_df.columns) /2, random_state = 7)
RFC.fit(X_train, y_train)
y_pred = RFC.predict(X_test)
RFC_probability = RFC.predict_proba(X_test)
print(f"TESTINFO Rnd F Cl: [{dt.today()}]--[Parameters: n_estimators:{RFC.n_estimators}, max_depth:{RFC.max_depth}, random state:{RFC.random_state}]--Training set accuracy: {RFC.score(X_train, y_train)}; Test set accuracy: {RFC.score(X_test, y_test)}; Test set validation: {RFC.score(X_test, y_pred)}")
#%%
'''
K Nearest Neighbor
'''
KNN = KNeighborsClassifier(n_neighbors = 8, weights = 'uniform',)
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)
print(f"TESTINFO KNN: [{dt.today()}]--[Parameters: n_neighbors:{KNN.n_neighbors}, weights:{KNN.weights}]--Training set accuracy: {KNN.score(X_train, y_train)}; Test set accuracy: {KNN.score(X_test, y_test)}; Test set validation: {KNN.score(X_test, y_pred)}")
#%%
'''
Use the random forest regressor algorithm to predict labels; DO NOT USE SCALED VARIABLES HERE
The number of splits for each tree level is equal to half the number of columns; that way overfitting is dampened and test remains fast
Test 4/22/2020: val_accuracy: 1.0 -> overfitted
'''
RFR = RandomForestRegressor(n_estimators = 75, max_depth = len(bank_df.columns)/2, min_samples_split = 4)
RFR.fit(X_train, y_train)
y_pred = RFR.predict(X_test)
print(f"TESTINFO Rnd F Reg: [{dt.today()}]--[Parameters: n_estimators:{RFR.n_estimators}, max_depth:{RFR.max_depth}, min_samples_split:{RFR.min_samples_split}]--Training set accuracy: {RFR.score(X_train, y_train)}; Test set accuracy: {RFR.score(X_test, y_test)}; Test set validation: {RFR.score(X_test, y_pred)}")
#%%
'''
                APPLICATION OF SKLEARN NEURAL NETWORK
works with minmax scaled version and  has very little accuracy depsite having 1000 layers
Test; [1000l;alpha=0.001] [2020-04-22 00:00]; Training set accuracy: 0.002171552660152009; Test set accuracy: 0.0
Test; [1500l; alpha=0.0001] [2020-04-22 14:16:43.290096]; Training set accuracy: 0.3517915309446254; Test set accuracy: 0.3468354430379747
'''
#adam: all-round solver for data
#hidden_layer_sizes: no. of nodes/no. of hidden weights used to obtain final weights;
#match with input features
#alpha: regularization parameter that shrinks weights toward 0 (the greater the stricter)
MLP = MLPClassifier(hidden_layer_sizes = 1500, solver='adam', alpha=0.0001 )
MLP.fit(X_train, y_train)
y_val = MLP.predict(X_test)
#y_val.reshape(-1, 1)
print(f"TESTINFO MLP: [{dt.today()}]--[Parameters: hidden layers:{MLP.hidden_layer_sizes}, alpha:{MLP.alpha}]--Training set accuracy: {MLP.score(X_train, y_train)}; Test set accuracy: {MLP.score(X_test, y_test)}")
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

features = np.array(X_train_scaled)
targets = np.array(y_train)
features_validation = np.array(X_test_scaled)
targets_validation = np.array(y_test)

print(features[:10])
print(targets[:10])
####
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
    hist = np.concatenate((hist, np.array(history.history['mean_squared_error'])), axis = 0)
#%%

#plot metrics
plt.plot(hist)
plt.show()

predictions = model.predict(features_validation, verbose=0)
print('R2 score = ', r2_score(y_test, predictions), '/ 1.0')
print('MSE score = ', mean_squared_error(targets_validation, predictions), '/ 0.0')
#######
plt.plot(features_validation.as_matrix()[0:50], '+', color ='blue', alpha=0.7)
plt.plot(predictions[0:50], 'ro', color ='red', alpha=0.5)
plt.show()
#%%
'''
                    APPLICATION OF TENSORFLOW
'''
#future needs to be run first
#eager execution needs to be run right after the TF instantiation to avoid errors
from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column
from tensorflow.keras import layers
#%%
#custom train test split
train, test = train_test_split(df, test_size = 0.2)
train, val = train_test_split(train, test_size = 0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
#%%
#create tuples for lists to organize the columns conversion
##Tuple (or list) for the bank list to refer to the length
#banks = list('Bank of America','Toronto Dominion Bank', 'Citizens Bank', 'Webster Bank',
#      'CHASE Bank', 'Citigroup', 'Capital One', 'HSBC Bank USA',
#      'State Street Corporation','MUFG Union Bank', 'Wells Fargo & Co.', 'Barclays',
#      'New York Community Bank', 'CIT Group', 'Santander Bank',
#      'Royal Bank of Scotland', 'First Rand Bank', 'Budapest Bank')

#trans_type = list('CorePro Deposit', 'CorePro Withdrawal', 'Internal CorePro Transfer',
#                   'Interest Paid', 'CorePro Recurring Withdrawal',
#                   'Manual Adjustment', 'Interest Adjustment')

##STEP 1
'''

'''
#feature columns to use in the layers
feature_columns_container = []

#numeric column needed in the model
#wrap all non-numerical columns with indicator col or embedding col
######IN V2 STATUS IS NUMERICAL; THATS WHY IT WILL THROW "CAST STRING TO FLOAT IS NOT SUPPORTED" ERROR######
#the argument default_value governs out of vocabulary values and how to replace them

for header in ['typeCode', 'amount', 'returnCode', 'CS_FICO_num']:
  feature_columns_container.append(feature_column.numeric_column(header))

#bucketized column

#categorical column with vocabulary list
#type_col = feature_column.categorical_column_with_vocabulary_list(
#        'type', ['CorePro Deposit',
#                 'CorePro Withdrawal',
#                 'Internal CorePro Transfer',
#                 'Interest Paid',
#                 'CorePro Recurring Withdrawal',
#                 'Manual Adjustment',
#                 'Interest Adjustment'])
#type_pos = feature_column.indicator_column(type_col)
#type_pos_2 = feature_column.embedding_column(type_col, dimension = 8)
#feature_columns_container.append(type_pos)
#feature_columns_container.append(type_pos_2)

#idea: words or fragments are a bucket and can be used to recognize recurring bills
#friendly_desc = feature_column.categorical_column_with_hash_bucket(
#        'friendlyDescription', hash_bucket_size = 2500)
#fr_desc_pos = feature_column.embedding_column(friendly_desc, dimension = 250)
#feature_columns_container.append(fr_desc_pos)

#created_date = feature_column.categorical_column_with_hash_bucket(
#        'createdDate', hash_bucket_size = 365)
#set the indicator column
#cr_d_pos = feature_column.indicator_column(created_date)
#feature_columns_container.append(cr_d_pos)

#entry = feature_column.categorical_column_with_vocabulary_list(
#        'isCredit', ['Y', 'N'])
#set the embedding column
#entry_pos = feature_column.embedding_column(entry, dimension = 3)
#feature_columns_container.append(entry_pos)

#ret_c = feature_column.categorical_column_with_vocabulary_list(
#        'returnCode', ['RGD', 'RTN', 'NSF'])

fee = feature_column.categorical_column_with_vocabulary_list('feeCode',
                                                             ['RGD',
                                                              'RTN',
                                                              'NSF'])
#set the embedding column
fee_pos = feature_column.embedding_column(fee, dimension = 3)
feature_columns_container.append(fee_pos)

#check = feature_column.categorical_column_with_vocabulary_list('check',
#                                                               ['Y', 'N'])
#set the indicator column
#check_pos = feature_column.embedding_column(check, dimension = 2)
#feature_columns_container.append(check_pos)

#acc_bal = feature_column.categorical_column_with_vocabulary_list('account_balance',
#                                                                 ['u100',
#                                                                  'o100u1000',
#                                                                  'o1000u10000',
#                                                                  'o10000'])
#set the indicator value
#acc_bal_pos = feature_column.embedding_column(acc_bal, dimension = 10)
#feature_columns_container.append(acc_bal_pos)


age = feature_column.bucketized_column(feature_column.numeric_column('Age'),
                                       boundaries = [18, 20, 22, 26, 31, 35])
feature_columns_container.append(age)


#cs_internal = feature_column.categorical_column_with_vocabulary_list('CS_internal',
#                                                                       ['Poor',
#                                                                        'Average',
#                                                                        'Excellent'])
#set the indicator value
#cs_positive = feature_column.embedding_column(cs_internal)
#feature_columns_container.append(cs_positive)

#FICO 700 is the initial score and also the average in the US
#The CS_FICO_num column is in this version converted to a bucketized column
#instead of passing it to the feature_column_container
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
def df_to_dataset(dataframe, shuffle = True, batch_size = 150):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Student')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size = len(dataframe))
  ds = ds.batch(batch_size)
  return ds
######OVERHAUL NEEDED HERE#####
#def make_input_fn(df):
#  def pandas_to_tf(pdcol):
    # convert the pandas column values to float
#    t = tf.constant(pdcol.astype('float32').values)
    # take the column which is of shape (N) and make it (N, 1)
#    return tf.expand_dims(t, -1)

#  def input_fn():
    # create features, columns
#    features = {k: pandas_to_tf(df[k]) for k in FEATURES}
#    labels = tf.constant(df[TARGET].values)
#    return features, labels
#  return input_fn

#def make_feature_cols():
#  input_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
#  return input_columns
##################################
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
# Check accuracy
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
#%%
