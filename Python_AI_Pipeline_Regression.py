# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:57:08 2020

@author: bill-
"""

'''
Pipeline function - Regression Pipeline
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
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline


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
from Python_eda_ai import split_data, pipeline_rfr, pipeline_sgd_reg, pipeline_trans_reg, pipeline_logreg, score_df
#%%

#CONNECTION TO FLASK/SQL
app = Flask(__name__)

#function can be bound to the script by adding a new URL
@app.route('/encoder')

    #SETTING THE ENVIRONMENT VARIABLE
    #$ set FLASK_APP=file_name.py
    #$ flask run
    # * Running on http://127.0.0.1:5000/

    ####COMMAND PROMPT#####
    #set env var in windows with name: FLASK_APP and value: path to app.py
    #switch in dev consolde to this path
    #C:\path\to\app>set FLASK_APP=hello.py

    #make the flask app listen to all public IPs
    #flask run --host=0.0.0.0

    #to enable all development features, like debugger and
    #automatic restart when code changes
    #set FLASK_ENV=development

    ##joint command to set env var and run the app
    #env FLASK_APP=Python_inc_exp_bal_database.py flask run


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
    #csv_export(df=bank_df, file_name='encoded_bank_dataframe')

    return bank_df

@app.route('/store_regression_pickle')

def store_pickle(file_name, model):

    """
    Usage of a Pickle Model -Storage of a trained Model
    """
    #specify file name in letter strings
    model_file = file_name
    with open(model_file, mode='wb') as m_f:
        pickle.dump(model, m_f)
    print(f"Model saved in: {os.getcwd()}")
    return model_file

@app.route('/open_regression_pickle')

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

'''STEP 1'''
bank_df = df_encoder(rng=4)

'''
STEP 2
SPLITTING UP THE DATA
'''
#drop target variable in feature df
#all remaining columns will be the features
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

'''
Principal Component Reduction
'''
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

#generate all gridsearch files and determine the accuracy
#grid_search_lr = pipeline_logreg()
#grid_search_sgd = pipeline_sgd_reg()
#grid_search_svr = pipeline_svr()

#best accuracy
#grid_search_rfr = pipeline_rfr()
'''STEP 2'''
grid_search_treg = pipeline_trans_reg()

#append objects to a list first
# pipeline_dictionary = {}

# estimators= [grid_search_lr, grid_search_sgd, grid_search_svr,
#              grid_search_rfr, grid_search_treg]

# for element in estimators:
#     print(element.best_score_)
#     #add accuracies to dictionary
#     regressor = element.estimator
#     accuracy = element.score(X_test, y_test)
#     best_accuracy= element.best_score_
#     #regressor as key and accuracies as tuple value
#     pipeline_dictionary[regressor].append((accuracy, best_accuracy))

'''STEP 3'''
logistic_regression_file = store_pickle(file_name="regression_model.sav", model=grid_search_treg)




