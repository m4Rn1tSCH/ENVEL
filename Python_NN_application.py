# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:34:29 2019

@author: bill-
"""

##loading packages
import pandas as pd
#installed separately in the conda base env and p37
import pandas_profiling
import numpy as np
import matplotlib as plt
import seaborn
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


#custom packages
#import Python_df_label_encoding
#%%
#converting the link
link = r"C:\Users\bill-\Desktop\Q_test_data_v2.csv"
link_1 = link.replace(os.sep, '/')
file = ''.join(link_1)
'''
This test data needs more rows than features that have been selected or various
functions might fail
'''
#loading the data frame
#first column is the index to avoid "unnamed column"
df = pd.read_csv(file, index_col = [0])
df.info()
###
##LOADING THE DATA FRAME CAN PRODUCE NAs WHEN ITS OPENED AND THE SCRIPT IS BEING RUN!!!!
#data type is object since the data frame mixes numerical and string columns

#'Student', 'account_balance', 'Age', 'CS_internal', 'CS_FICO_num', 'CS_FICO_str'
#these columns columns are added to the Q2 object
###
#%%
null_list = df.isnull().sum()
for x in null_list:
    if x > 0:
        print("There are NAs in the data!")
else:
    pass
#%%
#for jupyter-notebook; not practical here
#seaborn.pairplot(df)
#seaborn.heatmap(df)
#seaborn.clustermap(df)
#%%
###################APPLICATION OF LABELENCODER########################

#applying fit_transform yields: encoding of 22 columns but most of them remain int32 or int64
#applying first fit to train the data and then apply transform will encode only 11 columns and leaves the others unchanged
#if 2 or fewer unique categories data type changes to "object"
#iterate through columns and change the object (not int64)

###ATTENTION; WHEN THE DATA FRAME IS OPEN AND THE SCRIPT IS RUN
###THE DATA TYPES CHANGE TO FLOAT64 SINCE THE NUMBERS ARE BEING DISPLAYED
le = LabelEncoder()
le_count = 0
#           V1
#Iterate through the columns
#Train on the training data
#Transform both training and testing
#Keep track of how many columns were converted
for col in df:
    if df[col].dtype == 'object':
        le.fit(df[col])
        df[col] = le.transform(df[col])
        le_count += 1

print('%d columns were label encoded.' % le_count)

#for comparison of the old data frame and the new one
print("PROCESSED DATA FRAME:")
print(df.head(3))
print("new data frame ready for use")
####################################################################
#              V2
#for col in df:
#    if df[col].dtype == 'object':
#        df[col] = le.fit_transform(df[col])
#        le_count += 1

#print('%d columns were label encoded.' % le_count)

#for comparison of the old data frame and the new one
#print("PROCESSED DATA FRAME:")
#print(df.head(3))
#print("new data frame ready for use")
#####################################################################
#           V3
#col_list = list(df)
#create an empty data frame
#df_LE = pd.DataFrame()
#read all column headers as string to convert them properly
#for n in col_list:
#    ''.join(('', n, ''))
#    if col_list[col_list != 'int64'] and col_list[col_list != 'int64']:
#        df_LE[n] = le.fit_transform(df[n])
#print('%d columns were label encoded.' % le_count)

#for comparison of the old data frame and the new one
#print("PROCESSED DATA FRAME:")
#print(df.head(3))
#print("new data frame ready for use")
####################################################################
#%%
###################SPLITTING UP THE DATA###########################
#model_features = df
#model_label = df['Age']

#X_train, X_test, y_train, y_test = train_test_split(model_features, model_label,
#                                                    shuffle = True,
#                                                    test_size = 0.3)

#%%
#select statisitically significant features with Age as target variable
#chi2 for non-negative ONLY!!
#all features but one that becomes the label
features = len(df.columns) - 1
k_best = SelectKBest(score_func = chi2, k = 'all')
k_best.fit(df, df['Age'])
#optimal parameters picked
k_best.scores_
k_best.pvalues_
#%%
#select statisitically significant features with Student as target variable
#deduct 10 features from all available ones (20 left here)
#only the 20 best ones with the strongest correlation will be picked
features = len(df.columns) - 10
k_best = SelectKBest(score_func = chi2, k = features)
k_best.fit(df, df['Student'])
#optimal parameters picked
print(k_best.scores_)
print(k_best.pvalues_)

#%%
#############APPLICATION OF RECURSIVE FEATURE EXTRACTION/LOGISTIC REGRESSION###########################
#Creating training and testing data
train=df.sample(frac = 0.5,random_state = 200)
test=df.drop(train.index)

cols = ["type", "amount", "isCredit", "returnCode", "feeCode", "subTypeCode", "subType", "check", "Student", "account_balance", "Age", "CS_FICO_num", "CS_internal"]
X_train = train[cols]
y_train = train['CS_FICO_str']
X_test = test[cols]
y_test = test['CS_FICO_str']
# Build a logreg and compute the feature importances
log_reg = LogisticRegression()
# create the RFE model and select 8 attributes
rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
rfe = rfe.fit(X_train, y_train)
#selected attributes
print('Selected features: %s' % list(X_train.columns[rfe.support_]))

##Use the Cross-Validation function of the RFE modul
#accuracy describes the number of correct classifications
rfecv = RFECV(estimator = LogisticRegression(), step = 1, cv = 8, scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize = (10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
#%%
#############APPLICATION OF A RANDOM FOREST REGRESSOR##################
rfr = RandomForestRegressor()
#set up the parameters as a dictionary
parameters = {'n_estimators': [5, 10, 100],
              #'criterion': ['mse'],
              #'max_depth': [5, 10, 15],
              #'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1,5]
             }

#set up the GridCV
#cv determines the cross-validation splitting strategy /to specify the number of folds in a (Stratified)KFold
gridcv = GridSearchCV(estimator = rfr, param_grid = parameters,
                        cv = 5,
                        n_jobs = -1,
                        verbose = 1)

grid = gridcv.fit(X_train, y_train)

#activate the best combination of parameters
rfr = grid.best_estimator_
rfr.fit(X_train, y_train)
#%%
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

predictions = rfr.predict(X_test)

#if we want to Re-scale, use this lines of code :
#predictions = predictions * (max_train - min_train) + min_train
#y_validation_RF = y_validation * (max_train - min_train) + min_train

#if not, keep this one:
y_validation_RF = y_test

print('R2 score = ',r2_score(y_validation_RF, predictions), '/ 1.0')
print('MSE score = ',mean_squared_error(y_validation_RF, predictions), '/ 0.0')
#%%
###########################################################
'''
                        APPLICATION OF KERAS
'''
#features: X
#target: Y
import keras
features = np.array(X_train)
targets = np.array(y_train.values.reshape(y_train.shape[0],1))
features_validation = np.array(X_test)
targets_validation = np.array(y_test.values.reshape(y_test.shape[0],1))

print(features[:10])
print(targets[:10])
####
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

#building the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(1))

# Compiling the model
model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) #mse: mean_square_error
model.summary()

# Training the model
epochs_tot = 1000
epochs_step = 250
epochs_ratio = int(epochs_tot / epochs_step)
hist =np.array([])
#######
#building the epochs
for i in range(epochs_ratio):
    history = model.fit(features, targets, epochs=epochs_step, batch_size=100, verbose=0)

    # Evaluating the model on the training and testing set
    print("Step : " , i * epochs_step, "/", epochs_tot)
    score = model.evaluate(features, targets)
    print("Training MSE:", score[1])
    score = model.evaluate(features_validation, targets_validation)
    print("Validation MSE:", score[1], "\n")
    hist = np.concatenate((hist, np.array(history.history['mean_squared_error'])), axis = 0)

# plot metrics
plt.plot(hist)
plt.show()

predictions = model.predict(features_validation, verbose=0)
print('R2 score = ',r2_score(y_test, predictions), '/ 1.0')
print('MSE score = ',mean_squared_error(y_validation_RF, predictions), '/ 0.0')
#######
plt.plot(y_test.as_matrix()[0:50], '+', color ='blue', alpha=0.7)
plt.plot(predictions[0:50], 'ro', color ='red', alpha=0.5)
plt.show()
#%%
############APPLICATION OF SKLEARN NEURAL NETWORK#####################
'''
from sklearn.neural_network import MLP
mlp = MultiLayerPerceptron

'''
###########APPLICATION OF PYTORCH###############################

#%%
##############################################################
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
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
#%%
'''
This is an overview about the choice of columns for the model and how to preprocess them before
compiling the model and training it

Data columns (total 30 columns):
transactionCount       6 non-null int64
transactionId          6 non-null int64
masterId               6 non-null int64
customerId             6 non-null int64
type                   6 non-null object categorical (wrapped in embedding)x
typeCode               6 non-null int64 x
tag                    6 non-null int64
friendlyDescription    6 non-null int64 cat with hashed feat (wrapped in embedding) x
description            6 non-null int64
status                 6 non-null int64
createdDate            6 non-null object bucketized? x
amount                 6 non-null float64 x
isCredit               6 non-null object categorical (wrapped in embedding) x
settledDate            6 non-null object
availableDate          6 non-null object
voidedDate             6 non-null object
returnCode             6 non-null int64 categorical
feeCode                6 non-null int64 categorical (wrapped in embedding) x
feeDescription         6 non-null object
cardId                 6 non-null int64
subTypeCode            6 non-null object
subType                6 non-null object
institutionName        6 non-null object categorical (wrapped in indicator) x
check                  6 non-null object categorical (wrapped in embedding) x
Student                6 non-null int64 x
account_balance        6 non-null int64 bucketized (wrapped in embedding) x
Age                    6 non-null int64 bucketized x
CS_internal            6 non-null int64 bucketized/ categorical x
CS_FICO_num            6 non-null int64 categorical/bucketized if as strings (wrapped in embedding) x
CS_FICO_str            6 non-null int64 categorical
'''
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
#feature columns to use in the layers
feature_columns_container = []

#numeric column needed in the model
#other components altered to other data types
#wrap all non-numerical columns with indicator col or embedding col
for header in ['typeCode', 'status', 'amount']:
  feature_columns_container.append(feature_column.numeric_column(header))

#bucketized column

#categorical column with vocabulary list
type_col = feature_column.categorical_column_with_vocabulary_list(
        'type', ['CorePro Deposit',
                 'CorePro Withdrawal',
                 'Internal CorePro Transfer',
                 'Interest Paid',
                 'CorePro Recurring Withdrawal',
                 'Manual Adjustment',
                 'Interest Adjustment'])
type_col_pos = feature_column.embedding_column(type_col, dimension = 7)
feature_columns_container.append(type_col_pos)

#idea: words or fragments are a bucket and can be used to recognize recurring bills
friendly_desc = feature_column.categorical_column_with_hash_bucket(
        'friendlyDescription', hash_bucket_size = 2500)
fr_desc_pos = feature_column.embedding_column(friendly_desc, dimension = 250)
feature_columns_container.append(fr_desc_pos)

#created_date = feature_column.categorical_column_with_hash_bucket(
#        'createdDate', hash_bucket_size = 365)
#set the indicator column
#cr_d_pos = feature_column.indicator_column(created_date)
#feature_columns_container.append(cr_d_pos)

entry = feature_column.categorical_column_with_vocabulary_list(
        'isCredit', ['Y', 'N'])
#set the embedding column
entry_pos = feature_column.embedding_column(entry, dimension = 2)
feature_columns_container.append(entry_pos)

#ret_c = feature_column.categorical_column_with_vocabulary_list(
#        'returnCode', ['RGD', 'RTN', 'NSF'])

fee = feature_column.categorical_column_with_vocabulary_list('feeCode',
                                                             ['RGD',
                                                              'RTN',
                                                              'NSF'])
#set the embedding column
fee_pos = feature_column.embedding_column(fee, dimension = 3)
feature_columns_container.append(fee_pos)

check = feature_column.categorical_column_with_vocabulary_list('check',
                                                               ['Y', 'N'])
#set the indicator column
check_pos = feature_column.embedding_column(check, dimension = 2)
feature_columns_container.append(check_pos)

acc_bal = feature_column.categorical_column_with_vocabulary_list('account_balance',
                                                                 ['u100',
                                                                  'o100u1000',
                                                                  'o1000u10000',
                                                                  'o10000'])
#set the indicator value
acc_bal_pos = feature_column.embedding_column(acc_bal, dimension = 10)
feature_columns_container.append(acc_bal_pos)


age = feature_column.bucketized_column(feature_column.numeric_column('Age'),
                                       boundaries = [18, 20, 22, 26, 31, 35])
#set the indicator column
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
##########################
#%%
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle = True, batch_size = 32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Student')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size = len(dataframe))
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
  layers.Dense(units = 1, activation = 'sigmoid')
])
#%%
##STEP 5
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=2)
#%%
##STEP 6
# Check accuracy
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

##STEP 7
# Infer labels on a batch
predictions = model.predict(test_ds)