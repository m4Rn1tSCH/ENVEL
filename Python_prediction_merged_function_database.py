# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:49:25 2020

@author: bill-
"""
#load packages
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob

#from datetime import datetime
#import seaborn as sns
#plt.rcParams["figure.dpi"] = 600
#plt.rcParams['figure.figsize'] = [12, 10]
#%%
'''
Setup of the function to merge every single operation into one function that is then called by the flask connection/or SQL
contains: preprocessing, splitting, training and eventually prediction
'''
#CONNECTION TO FLASK/SQL
#INSERT FLASK CONNECTION SCRIPT HERE
###########################################
#loading the simplified applications
#from flask import Flask
#app = Flask(__Preprocessor__)

##put address here
#@app.route('/')
#def hello_world():
#    return 'Hello, World!'
#route tells what URL should trigger the function
#use __main__ only for the actual core of the application
# pick unique names for particular functions if these are imported later
#DONT CALL APPLICATIONS FLASK.PY TO AVOID CONFLICTS WITH FLASK

#RUN THE APPLICATION
#flask command or -m swith in Python

########SETTING THE ENVIRONMENT VARIABLE#######
#$ export FLASK_APP = C:\Users\bill-\OneDrive\Dokumente\Docs Bill\TA_files\functions_scripts_storage\Python_prediction_merged_function_database.py
#$ flask run
# * Running on http://127.0.0.1:5000/

####COMMAND PROMPT#####
#C:\path\to\app>set FLASK_APP=hello.py

####for production use##
#$ flask run --host=0.0.0.0
############################################
#INSERT SQL CONNECTION HERE
############################################
###SQL-CONNECTION TO QUERY THE VENDOR FILE
###Create engine
##engine = create_engine('sqlite:///Chinook.sqlite')

##Open engine connection: con
##con = engine.connect()

##Perform query: rs
##rs = con.execute("SELECT * from <<DB_FOLDER>>")

#Save results df
##df = pd.DataFrame(rs.fetchall())

##Close connection
##con.close()
##############################################
#%%
######LOADING THE TRANSACTION FILE#####
#transaction_file = r"C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx"
#path_1 = transaction_file.replace(os.sep,'/')
#transactions = ''.join(('', path_1, ''))
'''
The preprocessed files are CSV; the function will load all CSVs and pick label and features
for testing purposes:
'''
#relative_t_path = './*.csv'
########################
"C:\Users\bill-\Desktop\03-04-2020_CARD_PANEL.csv"
"C:\Users\bill-\Desktop\03-04-2020_BANK_PANEL.csv"
"C:\Users\bill-\Desktop\03-04-2020_DEMO_PANEL.csv"
#######################
#%%
#import files and append all directory paths to a list
basepath = 'C:/Users/bill-/Desktop/Harvard_Resumes'
path_list = []
#Walking a directory tree and printing the names of the directories and files
for dirpath, dirnames, filename in os.walk(basepath):
    print(f'Found directory: {dirpath}')
    for file in filename:
        if os.path.isfile(file):
            print("file found and appended")
        path_list.append(os.path.abspath(os.path.join(dirpath, file)))
#%%

#Write the pattern as a folder or file pattern
path_abs = os.path.abspath(os.path.join('C:/Users/bill-/Desktop/'))
pattern = '*.csv'
directory = os.path.join(path_abs, pattern)
#Save all file matches: csv_files
pdf_files = glob.glob(directory)
#Print the file names
#print(pdf_files)
#%%
def predict_needed_value(preprocessed_input):

    for file in pdf_files:
        df_file_rdy = pd.read_csv(file)
        print(f"dataframe {file} loaded and will be analyzed")
    #%%
    ##SELECTION OF FEATURES AND LABELS
    #first prediction loop and stop
    y = []
    X = []
    for col in list(df_card):
        if df_card[col].isnull().any() == True:
            print(f"{col} is target variable and will be used for prediction")
            y.append(df_card[col])
            if len(y) == 1:
                print("first prediction target found...")
                break
    #%%
    #PICK FEATURES AND LABELS
    #columns = list(df_card)
    #print("enter the label that is to be predicted...; all other columns will remain and picked later as features ranked by prediction importance")
    #label_str = input("What should be predicted?")
    #X = columns.pop(columns.index(label_str))

    #X = list(df_card)
    #set the label
    #y = list(df_card).pop(list(df_card)('amount'))
    #%%
    #APPLY THE SCALER FIRST AND THEN SPLIT INTO TEST AND TRAINING
    #PASS TO STANDARD SCALER TO PREPROCESS FOR PCA
    #ONLY APPLY SCALING TO X!!!
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    #fit_transform also separately callable; but this one is more time-efficient
    for col in X:
        X_scl = scaler.fit_transform(X)
    #%%
    #TRAIN TEST SPLIT INTO TWO DIFFERENT DATASETS
    #Train Size: percentage of the data set
    #Test Size: remaining percentage
    #from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_scl, y, test_size = 0.3, random_state = 42)
    #shape of the splits:

    ##features: X:[n_samples, n_features]
    ##label: y: [n_samples]

    print(f"Shape of the split training data set: X_train:{X_train.shape}")
    print(f"Shape of the split training data set: X_test: {X_test.shape}")
    print(f"Shape of the split training data set: y_train: {y_train.shape}")
    print(f"Shape of the split training data set: y_test: {y_test.shape}")
    #%%
        '''
    #COLUMNS df_card
    Index(['unique_mem_id', 'unique_card_account_id', 'unique_card_transaction_id',
           'amount', 'currency', 'description', 'transaction_date', 'post_date',
           'transaction_base_type', 'transaction_category_name',
           'primary_merchant_name', 'secondary_merchant_name', 'city', 'state',
           'zip_code', 'transaction_origin', 'factual_category', 'factual_id',
           'file_created_date', 'optimized_transaction_date',
           'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred', 'swipe_date',
           'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
           'account_type', 'account_source_type', 'account_score', 'user_score',
           'lag', 'is_duplicate'],
          dtype='object')
    #COLUMNS df_bank
    Index(['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id',
           'amount', 'currency', 'description', 'transaction_date', 'post_date',
           'transaction_base_type', 'transaction_category_name',
           'primary_merchant_name', 'secondary_merchant_name', 'city', 'state',
           'zip_code', 'transaction_origin', 'factual_category', 'factual_id',
           'file_created_date', 'optimized_transaction_date',
           'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred', 'swipe_date',
           'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
           'account_type', 'account_source_type', 'account_score', 'user_score',
           'lag', 'is_duplicate'],
          dtype='object')
    #COLUMNS df_demo
    Index(['unique_mem_id', 'state', 'city', 'zip_code', 'income_class',
           'file_created_date', 'yodlee_transaction_status', 'update_type',
           'panel_file_created_date'],
          dtype='object')
        '''
    ##
    #%%
    #PASS TO RECURSIVE FEATURE EXTRACTION
    '''
    all other columns are features and need to be checked for significance to be added to the feature list
    '''
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression

    #Creating training and testing data
    train = df_card.sample(frac = 0.5, random_state = 12)
    test = df_card.drop(train.index)

    #pick feature columns to predict the label
    #y_train/test is the target label that is to be predicted
    #PICKED LABEL = FICO numeric
    cols_card = ['unique_mem_id', 'unique_card_account_id', 'unique_card_transaction_id',
           'amount', 'currency', 'description', 'transaction_date', 'post_date',
           'transaction_base_type',# 'transaction_category_name',#
           'primary_merchant_name', 'secondary_merchant_name', 'city', 'state',
           'zip_code', 'transaction_origin', 'factual_category', 'factual_id',
           'file_created_date', 'optimized_transaction_date',
           'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred', 'swipe_date',
           'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
           'account_type', 'account_source_type', 'account_score', 'user_score',
           'lag', 'is_duplicate']
    X_train = df_card[cols_card]
    y_train = train['transaction_category_name']
    X_test = test[cols_card]
    y_test = test['transaction_category_name']
    #build a logistic regression and use recursive feature elimination to exclude trivial features
    log_reg = LogisticRegression()
    #create the RFE model and select the eight most striking attributes
    rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
    rfe = rfe.fit(X_train, y_train)
    #selected attributes
    print('Selected features: %s' % list(X_train.columns[rfe.support_]))
    print(rfe.ranking_)

    #Use the Cross-Validation function of the RFE module
    #accuracy describes the number of correct classifications
    rfecv = RFECV(estimator = LogisticRegression(), step = 1, cv = 8, scoring = 'accuracy')
    rfecv.fit(X_train, y_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

    #plot number of features VS. cross-validation scores
    #plt.figure(figsize = (10,6))
    #plt.xlabel("Number of features selected")
    #plt.ylabel("Cross validation score (nb of correct classifications)")
    #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #plt.show()