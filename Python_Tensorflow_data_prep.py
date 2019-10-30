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
#####
#index_col = None uses first column automatically
#index_col = False uses no index at all
data = pd.read_csv(file, skiprows = 1, index_col = False, na_values = 0,
                                                               names = ['date',
                                                                   'category',
                                                                   'trans_cat',
                                                                   'subcat',
                                                                   'shopname',
                                                                   'amount'])
#%%
#LabelEncoder for category, trans_cat, shopname
#use the LabelEncoder to make shopname numerical
LE = LabelEncoder()
data['LE_shopname'] = LE.fit_transform(data['shopname'])
#make sure the data is read as uniform data type; read everything as strings and encode them to numerical values!
data['LE_category'] = LE.fit_transform(data['category'].astype(str, copy = True, errors = 'raise'))
#CATEGORY HAS MIXED DATA TYPES; DTYPES 'OBJECT' INDICATES MIXED DATA TYPES!!!
#COMBINE LE FUNCTION FOR THAT AND ASTYPE MAKE SURE IT IS A UNIFORM DATATYPE

#fillna is obsolete; the LabelEncoder() removed all NaNs already
#category can be safely dropped as well since LE_category is now ready to be a feature
#do not create a new column but change the old one
#%%
#READY UP DATA FOR TENSORFLOW
data_features = data.drop(['date','category', 'trans_cat','subcat', 'shopname', 'LE_shopname'], axis = 1)
#DF READY
#convert it to an array to make it a feature
model_features = data_features.to_numpy(dtype = 'float32', copy = True)
#no labels to see if tensor can handle the input
model_label = data['LE_shopname'].to_numpy(dtype = 'float32', copy = True)
#PREDICTION MODEL: AMOUNT + LE_CATEGORY > LE_SHOPNAME
#%%
#INPUT: PANDAS DATA FRAMER
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
    #axis = None returns a static axis; axis = 1 returns a series or data frame
    random_sample = data.sample(n = sample_size, frac = None, replace = False, weights = sample_weights,
                             random_state = random_state, axis = 1)

    #ranking with index (axis = 0)
    #NA_option to either remove or keep the NaNs!
    ranked_sample = random_sample.rank(axis = 0, method = 'min', numeric_only = None, na_option = 'keep',
                                    ascending = True, pct = False)
    print(ranked_sample.head(3))

#%%
#train_test_split seems to take 2 times the identical data frame
##FEATURES
#LE_category       int32
#amount          float64
##LABEL
#LE_shopname       int32

#Equation: amount + LE_category > LE_shopname

#splitting of the data
#look up structure of train test split; since x= and y=  doesnt work sometimes
X_train, X_test, y_train, y_test = train_test_split(model_features, model_label, test_size = 0.5)
#shape of the splits:
#features: X:[n_samples, n_features]
#label: y: [n_samples]
print(f"Shape of the split training data set: X_train:{X_train.shape}")
print(f"Shape of the split training data set: X_test: {X_test.shape}")
print(f"Shape of the split training data set: y_train: {y_train.shape}")
print(f"Shape of the split training data set: y_test: {y_test.shape}")
#%%
#feature selection for artificial data in Q2 format
#default for number of features picked is 10!
#set no. of features picked to no. of columns - 1
#that way all features will be taken into consideraton to predict one label
#ENCAPSULATE 'ALL' IN LETTER TICKS TO AVOID MISINTERPRETATIONS IN THE FUNCTIONS
from sklearn.feature_selection import SelectKBest, f_classif
features = len(data.columns) - 1
k_best = SelectKBest(score_func = f_classif, k = 'all')
k_best.fit(X_train, y_train)
#optimal parameters picked
#%%
#CLASSIFICATION & REGRESION BELOW
#GRADIENT BOOSTING REGRESSOR
#depth shouldnt allow overfitting; keep smaller than number of features available
#CANNOT TAKE DATA WITH NAN, INFINITY OR FLOAT 32
from sklearn.ensemble import GradientBoostingRegressor
#alpha: regularization parameter; the higher, the stricter the parameters are forced toward zero
GBR = GradientBoostingRegressor(alpha = 0.05, learning_rate = 0.05, n_estimators = 150, max_depth = 5 , random_state = 0)
GBR.fit(X_train, y_train)
y_test = GBR.predict(X_test)
f"Training set accuracy: {GBR.score(X_train, y_train)}; Test set accuracy: {GBR.score(X_test, y_test)}"
#%%
#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(max_depth = 5, max_features = 'auto', n_estimators = 25, random_state = None, n_jobs = -1)
RFC.fit(X_train, y_train)
y_test = RFC.predict(X_test)
f"Training set accuracy: {RFC.score(X_train, y_train)}; Test set accuracy: {RFC.score(X_test, y_test)}"
#%%
#RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(max_depth = 5, max_features = 'auto', n_estimators = 25, random_state = None, n_jobs = -1)
RFR.fit(X_train, y_train)
y_test = RFR.predict(X_test)
f"Training set accuracy: {RFR.score(X_train, y_train)}; Test set accuracy: {RFR.score(X_test, y_test)}"
#%%
#Ridge Regression
from sklearn.linear_model import Ridge
Ridge = Ridge(alpha = 1.0, random_state = None)
Ridge.fit(X_train, y_train)
y_test = Ridge.predict(X_test)
f"Training set accuracy: {Ridge.score(X_train, y_train)}; Test set accuracy: {Ridge.score(X_test, y_test)}"
#%%
#Neural Network
#NO GPU SUPPORT FOR SKLEARN
from sklearn.neural_network import MLPClassifier

#adam: all-round solver for data
#hidden_layer_sizes: no. of nodes/no. of hidden weights used to obtain final weights; match with input features
#alpha: regularization parameter that shrinks weights toward 0 (the greater the stricter)
MLP = MLPClassifier(hidden_layer_sizes = 100, solver='adam', alpha=0.01 )
MLP.fit(X_train, y_train)
f"Training set accuracy: {MLP.score(X_train, y_train)}; Test set accuracy: {MLP.score(X_test, y_test)}"
#%%
#K Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
metric_params = None, n_jobs = 1, n_neighbors = 1, p = 2, weights = 'uniform')
KNN.fit(X_train, y_train)
y_test = KNN.predict(X_test)
f"Training set accuracy: {KNN.score(X_train, y_train)}; Test set accuracy: {KNN.score(X_test, y_test)}"