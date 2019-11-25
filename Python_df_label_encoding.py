# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:34:01 2019

@author: bill-
"""

#load the required packages
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd

def label_encoding(data_path):

    #converting the link
    link = data_path
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
    #read all column headers as  string to convert them properly
    for n in col_list:
        ''.join(('', n, ''))
        if col_list[col_list != 'int64']:
            df_LE[n] = le.fit_transform(df[n].astype(str,
                                                 copy = True,
                                                 errors = 'raise'))
            df = pd.concat([df, df_LE], axis = 1)
    else:
        pass

#for comparison of the old data frame and the new one
    print("NEW DATA FRAME:")
    print(df_LE.head(3))
    print("new data frame ready for use ")
    print("Name: df_LE")