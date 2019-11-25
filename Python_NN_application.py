# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:34:29 2019

@author: bill-
"""

##loading packages
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2

#custom packages
#import Python_df_label_encoding

#converting the link
link = r"C:\Users\bill-\Desktop\Q_test_data.csv"
link_1 = link.replace(os.sep, '/')
file = ''.join(link_1)

#loading the data frame
#first column is the index
df = pd.read_csv(file, index_col = 0)
###
##LOADING THE DATA FRAME CAN PRODUCE NAs WHEN ITS OPENED AND THE SCRIPT IS BEING RUN!!!!
#data type is object since the data frame mixes numerical and string columns

#'Student', 'account_balance', 'Age', 'CS_internal', 'CS_FICO_num', 'CS_FICO_str'
#these columns columns are added to the Q2 object
#applying the custom function
###
#%%
null_list = df.isnull().sum()
for x in null_list:
    if x > 0:
        print("There are NAs in the data!")
else:
    pass
#%%
###################APPLICATION OF LABELENCODER########################
#instantiate the label encoder
le = LabelEncoder()
#%%
#produce a iterable list of strings with all the columns
col_list = list(df)
#create an empty data frame
df_LE = pd.DataFrame()
df_2 = pd.DataFrame()
#%%
print("OLD DATA FRAME:")
print(df.head(3))
#%%
#iterate through columns and change the object (not int64)
#read all column headers as  string to convert them properly
###ATTENTION; WHEN THE DATA FRAME IS OPEN AND THE SCRIPT IS RUN THE DATA TYPES CHANGES TO FLOAT64
for n in col_list:
    ''.join(('', n, ''))
    if col_list[col_list != 'int64'] and col_list[col_list != 'float64']:
        df_LE[n] = le.fit_transform(df[n])
        df_2 = pd.concat([df, df_LE], axis = 1)
else:
    pass

#for comparison of the old data frame and the new one
print("NEW DATA FRAME:")
print(df_LE.head(3))
print("new data frame ready for use\n")
print("Name: df_LE")
#%%
#this module will turn every column into a int32 column
#will also convert all IDs and Numbers

#model_features =
#model_label =

#X_train, X_test, y_train, y_test = train_test_split(model_features, model_label,
#                                                    test_size = 0.3)

#%%
#select statisitically significant features with Age as target variable
from sklearn.feature_selection import SelectKBest, f_classif
#all features but one that becomes the labels
features = len(df.columns) - 1
k_best = SelectKBest(score_func = f_classif, k = 'all')
k_best.fit(df_LE, df_LE['Age'])
#optimal parameters picked
k_best.scores_
k_best.pvalues_
#%%
############APPLICATION OF SKLEARN NEURAL NETWORK#####################
from sklearn.neural_network import MLP


###########APPLICATION OF PYTORCH###############################


#############APPLICATION OF TENSORFLOW##########################