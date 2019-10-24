# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:01:09 2019

@author: bill-
"""

#load the transaction data and all packages
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#%%
#switching the backslashes to slashes
link = r"C:\Users\bill-\Desktop\TransactionsD.csv"
new_link = link.replace(os.sep, '/')
file = ''.join(('', new_link,''))

#load the data and skip the first row, then rename the columns to something informative
#columns
#date = date of transaction
#trans_cat = category of transaction
#subcat = subcategory
#shopname = shop name
#amount = amount in USD
#index_col = None uses first column automatically
#index_col = False uses no index at all
data = pd.read_csv(file, skiprows = 1, index_col = False, names = ['date', 'category', 'trans_cat', 'subcat', 'shopname', 'amount'])
df_no_date = data.drop(labels = ['date'], axis = 1)
#%%
#LabelEncoder for category, trans_cat, shopname
#use the LabelEncoder to make shopname numerical
LE = LabelEncoder()
data['LE_shopname'] = LE.fit_transform(data['shopname'])
#category does not support conversion of strings; is being converted to numbers and then to integers values
#do not create a new column but change the old one
data['category'] = data['category'].replace(to_replace={'Shops': '1',
                                                        'Food and Drink': '2',
                                                        'Travel': '3',
                                                        'Service': '4',
                                                        'Transfer': '5',
                                                        'Community': '6',
                                                        'Bank Fees': '7',
                                                        'Recreation': '8'},
                                                        value = None)
#fill in category not available values with 0
data['category'].fillna(value = 0)
#change to categorical data
data['category'] = pd.Categorical(data['category'])
#change to float number for potential preprocessing
data['category'].astype('float64')
#READY UP DATA TO BE READY FOR FEATURES AND PASS IT TO TENSOR FLOW
data_features = data.drop(labels = ['date', 'trans_cat', 'shopname', 'LE_shopname'], axis = 1)
#convert it to an array to make it a feature
model_features = data_features.to_numpy()
#no labels to see if tensor can handle the input
model_label = data['LE_shopname'].to_numpy()
#%%
#INPUT: PANDAS DATA FRAME
#OUTPUT: OBJECT TYPE THAT CANT BE USED FOR FURTHER OPERATIONS IF IT IS NOT CONVERTED TO A DATA FRAME AGAIN

#create a random sample
#picking 32 rows randomly (not subsequent ones) from the data and ranking it by date in ascending order (long ago to recent)
#set number of drawn rows/columns, optionally set a weight and a reproducible pseudo-random result
# axis in the rank method is set to None and returns a stat axis; change to 1 to return a data frame

##DATA = PANDAS DATA FRAME
##GIVE INTEGERS ONLY
sample_size = 32
sample_weights = None
random_state = None

def draw_sample(data, sample_size, sample_weights, random_state):
    #draw the sample and rank it
    #axis = None returns a stat axis; axis = 1 returns a series or data frame
    random_sample = data.sample(n = sample_size, frac = None, replace = False, weights = sample_weights,
                             random_state = random_state, axis = 1)

    #ranking with index (axis = 0)
    ranked_sample = random_sample.rank(axis = 0, method = 'min', numeric_only = None, na_option = 'keep',
                                    ascending = True, pct = False)
    print(ranked_sample.head(3))

#%%
#train_test_split seems to take 2 times the identical data frame
#DATA_FEATURES=LOADED TRANSACTION DATA
#category       category
#subcat          float64
#amount          float64
#LE_shopname       int32

#Equation: category + subcategory + amount ~ LE_shopname

#splitting of the data
#look up structure of train test split; since x= and y=  doesnt work sometimes
X_train, X_test, y_train, y_test = train_test_split(model_features,model_label, test_size = 0.3)
#%%
#data
#preprocess
#draw a sample
#result is a none type object
#convert to data frame
#nodate_df as feat
#regular df as label
#adjust random state to obtain identical samples that align within the data frame
#random_state = 42
#df_no_date_sample = draw_sample(df_no_date, sample_size = 32, sample_weights = None, random_state = 42)
#sample_df_nodate = pd.DataFrame(data = df_no_date_sample)
##
#data_sample = draw_sample(data, sample_size = 32, sample_weights = None, random_state = 42)
#sample_df = pd.DataFrame(data = data_sample)
##train test split possible with this set
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x = df_no_date_sample, y = data_sample, test_size = 0.4)
#%%
#classification & regresion below
#Gradient Boosting Regressor
#depth shouldnt allow overfitting; keep smaller than number of features available
from sklearn.ensemble import GradientBoostingRegressor
#alpha: regularization parameter; the higher, the stricter the parameters are forced toward zero
GBR = GradientBoostingRegressor(alpha = 0.05,learning_rate = 0.05, n_estimators = 150,max_depth = 5 ,random_state = 0)
GBR.fit(X_train, y_train)