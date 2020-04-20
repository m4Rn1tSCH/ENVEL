# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:51:04 2020

@author: bill-
"""
'''
EDA module for various Yodlee dataframes
FIRST STAGE: retrieve the user ID dataframe with all user IDs with given filter
SECOND STAGE: randomly pick a user ID; encode thoroughly and yield the df
'''

#load needed packages
import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np
from datetime import datetime as dt
#from flask import Flask
import os
import csv
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime as dt

from sklearn.feature_selection import SelectKBest , chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

#imported custom function
#generates a CSV for daily/weekly/monthly account throughput; expenses and income
from Python_spending_report_csv_export_function import spending_report
#contains the connection script
from Python_SQL_connection import execute_read_query, create_connection, close_connection
#contains all credentials
import PostgreSQL_credentials as acc
#%%
def df_preprocessor(rng = 2):
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
    The Counter object keeps track of permutations in a dictionary which can ten be read and
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
    This report measures either the sum orm ean of transactions happening
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
    spending_report(df = bank_df.copy())
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
    The lagging features produce NaN for the very first rows due to unavailability
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

    #bank_df.set_index("transaction_date", drop = False, inplace = True)
    #yield bank_df

#%%
#the first two rows of algging values have NaNs which need to be dropped
#drop the first and second row
bank_df = bank_df.drop([0, 1])
bank_df.reset_index(drop = True, inplace = True)
#%%
#BUGGED
#this squares the entire df and gets rid of non-negative values;
#chi2 should be applicable
# df_sqr = bank_df.copy()
# for col in df_sqr:
#     if df_sqr[col].dtype == 'int32' or df_sqr[col].dtype == 'float64':
#         df_sqr[col].apply(lambda x: np.square(x))
#%%
###################SPLITTING UP THE DATA###########################
#drop target variable in feature df
#all remaining columns will be the features
bank_df = bank_df.drop(['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id'], axis = 1)
model_features = np.array(bank_df.drop(['primary_merchant_name'], axis = 1))
model_label = np.array(bank_df['primary_merchant_name'])

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
#BUGGED
#STD SCALING - does not work yet
#fit the scaler to the training data first
#standard scaler works only with maximum 2 dimensions
scaler_obj = StandardScaler(copy = True, with_mean = True, with_std = True).fit(X_train)
X_train_scaled = scaler_obj.transform(X_train)

scaler_obj.mean_
scaler_obj.scale_
#transform data in the same way learned from the training data
X_test_scaled = scaler_obj.transform(X_test)
#%%
#MINMAX SCALING - works with Select K Best
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)

X_test_minmax = min_max_scaler.transform(X_test)
min_max_scaler.scale_
min_max_scaler.min_
#%%
#f_classif for regression
#chi-sqr for classification but requires non-neg values
####syntax of reshape(n_samples, n_features)
#### value of -1 allows for adaptation to shape needed
y_train_rs = np.array(y_train).reshape(-1, 1)
X_train_scl_rs = np.array(X_train_scaled).reshape(-1, 1)
X_test_scl_rs = np.array(X_test_scaled).reshape(-1, 1)

X_train_minmax_rs = X_train_minmax.reshape(-1, 1)
X_test_minmax_rs = X_test_minmax.reshape(-1, 1)
#%%
#fed variables cannot have missing values
'''
takes unscaled numerical so far and minmax scaled arguments
'''
k_best = SelectKBest(score_func = f_classif, k = 10)
k_best.fit(X_train_minmax, y_train)
k_best.get_params()

# isCredit_num = [1 if x == 'Y' else 0 for x in isCredits]
# np.corrcoef(np.array(isCredit_num), amounts)
#%%
#BUGGED
#pick feature columns to predict the label
#y_train/test is the target label that is to be predicted

cols = [c for c in bank_df if bank_df[c].dtype == 'int64' or 'float64']
X_train = bank_df[cols].drop(columns = ['primary_merchant_name'], axis = 1)
y_train = bank_df['primary_merchant_name']
X_test = bank_df[cols].drop(columns = ['primary_merchant_name'], axis = 1)
y_test = bank_df['primary_merchant_name']

#build a logistic regression and use recursive feature elimination to exclude trivial features
log_reg = LogisticRegression()
# create the RFE model and select the eight most striking attributes
rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
rfe = rfe.fit(X_train_minmax, y_train)
#selected attributes
print('Selected features: %s' % list(X_train_minmax.columns[rfe.support_]))
print(rfe.ranking_)
#%%
#BUGGED
#Use the Cross-Validation function of the RFE modul
#accuracy describes the number of correct classifications
rfecv = RFECV(estimator = LogisticRegression(), step = 1, cv = None, scoring='accuracy')
rfecv.fit(X_train_minmax, y_train)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train_minmax.columns[rfecv.support_]))

#plot number of features VS. cross-validation scores
# plt.figure(figsize = (10,6))
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()
#%%
#SelectKBest picks features based on their f-value to find the features that can optimally predict the labels
#funtion of Select K Best is here f_classifier; determines features based on the f-values between features & labels
#other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression



#Create pipeline with feature selector and classifier
#replace with gradient boosted at this point or regressor
pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func = f_classif)),
    ('clf', GradientBoostingClassifier(random_state = 42))])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERs NEED TO HAVE THE SAME LENGTH
params = {
    'feature_selection__k':[1, 2, 3, 4, 5, 6, 7],
    'clf__n_estimators':[15, 25, 50, 75, 120, 200, 350]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)
#%%
'''
                APPLICATION OF SKLEARN NEURAL NETWORK
works with minmax scaled version and  has very little accuracy depsite having 1000 layers
Training set accuracy: 0.002171552660152009; Test set accuracy: 0.0
'''
#adam: all-round solver for data
#hidden_layer_sizes: no. of nodes/no. of hidden weights used to obtain final weights;
#match with input features
#alpha: regularization parameter that shrinks weights toward 0 (the greater the stricter)
MLP = MLPClassifier(hidden_layer_sizes = 1000, solver='adam', alpha=0.001 )
MLP.fit(X_train_minmax, y_train)
y_val = MLP.predict(X_test)
#y_val.reshape(-1, 1)
print(f"Training set accuracy: {MLP.score(X_train, y_train)}; Test set accuracy: {MLP.score(X_test, y_test)}")
