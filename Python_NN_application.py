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
import Python_df_label_encoding

#converting the link
link = r"C:\Users\bill-\Desktop\Q_test_data.csv"
link_1 = link.replace(os.sep, '/')
file = ''.join(link_1)

#loading the data frame
#first column is the index
df = pd.read_csv(file, index_col = 0)
#data type is object since the data frame mixes numerical and string columns

#'Student', 'account_balance', 'Age', 'CS_internal', 'CS_FICO_num', 'CS_FICO_str'
#these columns columns are added to the Q2 object

#all data types of all columns
print(df.dtypes)

###################APPLICATION OF LABELENCODER########################
#%%
#this module will turn every columns into a int32 column
#will also convert all IDs and Numbers

############APPLICATION OF SKLEARN NEURAL NETWORK#####################


###########APPLICATION OF PYTORCH###############################


#############APPLICATION OF TENSORFLOW##########################