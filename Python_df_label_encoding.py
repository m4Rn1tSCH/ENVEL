# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:34:01 2019

@author: bill-
"""

#load the required packages
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd

#converting the link
link = r"C:\Users\bill-\Desktop\Q_test_data.csv"
link_1 = link.replace(os.sep, '/')
file = ''.join(link_1)
#%%
#loading the data frame
#first column is the index
df = pd.read_csv(file, index_col = 0)
#%%
#instantiate the label encoder
le = LabelEncoder()
#%%
#produce a iterable list of strings with all the columns
col_list = list(df)
#create an empty data frame
df_LE = pd.DataFrame()
#%%
print("OLD DATA FRAME:")
print(df.head(3))
#%%
#iterate through columns and change the object (not int64)
for n in col_list:
    ''.join(('', n, ''))
    if col_list[col_list != 'int64']:
        df_LE[n] = le.fit_transform(df[n].astype(str,
                                             copy = True,
                                             errors = 'raise'))
    #dfLE = df.append(n)
else:
    pass

#%%
print("NEW DATA FRAME:")
print(df_LE.head(3))
print("new data frame ready for use ")
print("Name: df_LE")