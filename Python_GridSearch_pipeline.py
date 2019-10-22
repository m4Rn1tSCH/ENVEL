#!/usr/bin/env python
# coding: utf-8
#Following classifier initiated with following values;
#
#Packages
import pandas as pd
import numpy as np
import os

#%%
#link conversion
link = r"C:\Users\bill-\Desktop\TransactionsD.csv"
new_link = link.replace(os.sep, '/')
file = ''.join(('', new_link,''))
#%%
#load the data and skip the first row, then rename the columns
#columns
#date = date of transaction
#trans_cat = category of transaction
#subcat = subcategory
#shopname = shop name
#amount = amount in USD
data = pd.read_csv(file, skiprows = 1, index_col = None, names =
                   ['category', 'trans_cat', 'subcat', 'shopname', 'amount'])
#%%
#use the LabelEncoder to make shopname numerical

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data['feat_shopname'] = LE.fit_transform(data['shopname'])
data.dtypes
#%%
#create numerical values
data_1 = data.replace(to_replace={'Shops': '1',
                                  'Food and Drink': '2',
                                  'Travel': '3',
                                  'Service': '4',
                                  'Transfer': '5',
                                  'Community': '6',
                                  'Bank Fees': '7',
                                  'Recreation': '8'},
                                    value=None)

#%%
#ready up labels and features for train and test split
#make sure everything is float or integer
#remove NAs to work with regressors (replaced with 0)
data_features = pd.concat([data['feat_shopname'], data['amount']], axis = 1)
data_label = data_1['category'].astype('float64').fillna(value = 0)
#%%
#split into 2 different data sets
#FEATURES: feat_shopname(int32) + amount(float64)
#LABELS: category(float64)
#Train Size: 50% of the data set
#Test Size: remaining 50%
from sklearn.model_selection import train_test_split
X = data_features
y = data_label
#split with 50-50 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#%%
from sklearn.pipeline import Pipeline
# Create pipeline with feature selector and classifier
pipe = Pipeline([
    ('feature_selection', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier(random_state=2))])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERs NEED TO HAVE THE SAME LENGTH
params = {
   'feature_selection__k':[2, 3, 4, 5],
   'clf__n_estimators':[20, 50, 75, 150]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid=params)

#Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)
