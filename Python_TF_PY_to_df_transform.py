# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:43:34 2019

@author: bill-
"""

import pandas as pd
import os

#get the data
link = r"C:\Users\bill-\Desktop\TransactionsD.csv"
new_link = link.replace(os.sep, '/')
file = ''.join(('', new_link,''))

#load the data
#columns
#date = date of transaction
#trans_cat = category of transaction
#subcat = subcategory
#shopname = shop name
#amount = amount in USD
data = pd.read_csv(file, skiprows = 1, index_col = None,
                   names = ['category',
                            'trans_cat',
                            'subcat',
                            'shopname',
                            'amount'])
#%%
#load all packages in sklearn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split
#%%
##################FUNCTION#####################
##preprocess all columns of a data frame to prepare it for Tensorflow/PYtorch
#and bring it a numpy array form
def tensor_pytorch_conversion (file):
    #print out a list of the column names
    column_list = file.columns
    #for each element of the column list apply the Label Encoder
    try:
        for col in column_list:
            file.i = le.fit_transform(file.i)
            #produce an empty list to append columns to drop
            drop_list = []
            if file.dtypes[file.dtypes != 'int64'] and  file.dytpes[file.dtypes != 'float64']:
                drop_list.append(column_list[col])
            else:
                #pass statement is an empty function/loop
                #is used to add things that are unfinished
                pass
    except:
        print("Conversion failed; \n data frame is mixed and cannot be used for ML in Tensor in PYtorch")
        return
    finally:
        file.drop([drop_list], axis = 1)
        file.to_numpy(dtype = 'float32', copy = True)
#%%
if (file.dtypes[file.dtypes == 'float64']):
    print("conversion of data successful")
    print(file.dtypes)
else:
    print("conversion failed...")
#%%
#conduct split
