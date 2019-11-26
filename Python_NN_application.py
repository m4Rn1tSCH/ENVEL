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
link = r"C:\Users\bill-\Desktop\Q_test_data.csv"
link_1 = link.replace(os.sep, '/')
file = ''.join(link_1)

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
#import seaborn
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

###ATTENTION; WHEN THE DATA FRAME IS OPEN AND THE SCRIPT IS RUN THE DATA TYPES CHANGE TO FLOAT64
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
model_features = df
model_label = df['Age']

X_train, X_test, y_train, y_test = train_test_split(model_features, model_label,
                                                    shuffle = True,
                                                    test_size = 0.3)

#%%
#select statisitically significant features with Age as target variable
from sklearn.feature_selection import SelectKBest, chi2
#all features but one that becomes the label
features = len(df.columns) - 1
k_best = SelectKBest(score_func = chi2, k = 'all')
k_best.fit(df, df['Age'])
#optimal parameters picked
k_best.scores_
k_best.pvalues_
#%%
#select statisitically significant features with Student as target variable
from sklearn.feature_selection import SelectKBest, chi2
#all features but one that becomes the labels
features = len(df.columns) - 1
k_best = SelectKBest(score_func = chi2, k = 'all')
k_best.fit(df, df['Student'])
#optimal parameters picked
print(k_best.scores_)
print(k_best.pvalues_)
#%%
#############APPLICATION OF RIDGE REGRESSION###########################
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
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=8, scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
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
############APPLICATION OF KERAS#############################
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
from sklearn.neural_network import MLP
mlp = MultiLayerPerceptron

###########APPLICATION OF PYTORCH###############################


#############APPLICATION OF TENSORFLOW##########################