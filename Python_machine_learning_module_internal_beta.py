# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:05:14 2020

@author: bill-
"""

#############ML logic for internal beta period#########
#INCOMING TRANSACTION
'''
Incoming approved transaction transmitted by the Q2 API in the Q2-format
'''
###connection to flask with relative link#####
###FLASK FUNCTION###
from flask import Flask
####PACKAGES OF THE MODULE######
import pandas as pd
import os
import re
##from sqlalchemy import create_engine
############################################
###SQL-CONNECTION TO QUERY THE VENDOR FILE
###Create engine named engine
##engine = create_engine('sqlite:///Chinook.sqlite')

##Open engine connection: con
##con = engine.connect()

##Perform query: rs
##rs = con.execute("SELECT * from <<DB_FOLDER>>")

#Save results of the query to DataFrame: df
##df = pd.DataFrame(rs.fetchall())

##Close connection
##con.close()
##############################################
#%%
######LOADING THE TWO FILES#####
#transaction_file = r"C:\Users\bill-\Desktop\TransactionsD_test.csv"
#path_1 = transaction_file.replace(os.sep,'/')
#transactions = ''.join(('', path_1, ''))
relative_t_path = './TransactionsD_test.csv'
#%%
#CHECK FOR TRANSACTION HISTORY
'''
If previous transactions are in the list/database/history, it is passed to check for missing values
If htere are no precedent transactions, the object will be passed to Paavna's splitting algorithm
'''
#%%
#PASS TO PAAVNA'S ALGORITHM
'''
Paavna's splitting algorithm will apply an initial split without the involvement of AI or a dynamic element
'''
#%%
#CHECK FOR MISSING VALUES
'''
find missing values and mark the corresponding column as target that is to be predicted
'''
#%%
#LABEL ENCODER
'''
encode all non-numerical values to ready up the data set for classification and regression purposes
'''
#applying fit_transform yields: encoding of 22 columns but most of them remain int32 or int64
#applying first fit to train the data and then apply transform will encode only 11 columns and leaves the others unchanged
#if 2 or fewer unique categories data type changes to "object"
#iterate through columns and change the object (not int64)

###ATTENTION; WHEN THE DATA FRAME IS OPEN AND THE SCRIPT IS RUN
###THE DATA TYPES CHANGE TO FLOAT64 AS THE NUMBERS ARE BEING DISPLAYED
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
#%%
#PASS TO RECURSIVE FEATURE EXTRACTION
'''
all other columns are features and need to be checked for significance to be added to the feature list
'''
#%%
#PASS TO ML ALGO
#RANDOM FOREST MODULE
'''
use the random forest algorithm to predict labels
'''
#Packages
import pandas as pd
import os

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
#split into 2 different data sets
#FEATURES: feat_shopname(int32) + amount(float64)
#LABELS: category(float64)
#Train Size: 50% of the data set
#Test Size: remaining 50%
from sklearn.model_selection import train_test_split
X = data_features
y = data_label
#split with 50-50 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)
#shape of the splits:
#features: X:[n_samples, n_features]
#label: y: [n_samples]
print(f"Shape of the split training data set: X_train:{X_train.shape}")
print(f"Shape of the split training data set: X_test: {X_test.shape}")
print(f"Shape of the split training data set: y_train: {y_train.shape}")
print(f"Shape of the split training data set: y_test: {y_test.shape}")
#%%
#SelectKBest picks features based on their f-value to find the features that can optimally predict the labels
#funtion of Selecr K Best is here f_classifier; determines features based on the f-values between features & labels
#other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression
from sklearn.feature_selection import SelectKBest, f_classif
#RandomForestClassifier is insufficient and does not provide enough splits
from sklearn.ensemble import RandomForestClassifier
#%%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# Create pipeline with feature selector and classifier
#replace with gradient boosted at this point or regressor
pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func = f_classif)),
    ('clf', RandomForestClassifier(random_state=2))])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERS NEED TO HAVE THE SAME LENGTH
params = {
   'feature_selection__k':[1, 2],
   'clf__n_estimators':[20, 50, 75, 150]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)
#%%
#GRADIENT BOOSTING DECISION TREE
'''
use the gradient boosting decision to predict labels
'''
#Train Size: 50% of the data set
#Test Size: remaining 50%
from sklearn.model_selection import train_test_split
X = data_features
y = data_label
#split with 50-50 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#SelectKBest picks features based on their f-value to find the features that can optimally predict the labels
#funtion of Selecr K Best is here f_classifier; determines features based on the f-values between features & labels
#other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# Create pipeline with feature selector and classifier
#replace with gradient boosted at this point or regressor
pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func = f_classif)),
    ('clf', GradientBoostingClassifier(random_state=42))])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERs NEED TO HAVE THE SAME LENGTH
params = {
   'feature_selection__k':[1, 2],
   'clf__n_estimators':[20, 50, 75, 150]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid=params)

#Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)
##RESULT
#the labels only provide one member per class, that makes the current data set
#unsuitable for a pickle file
#%%
#KNN
'''
use the K-nearest neighbor to predict labels
'''
#K NEAREST NEIGHBOR CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
metric_params = None, n_jobs = 1, n_neighbors = 1, p = 2, weights = 'uniform')
KNN.fit(X_train, y_train)
y_test = KNN.predict(X_test)
f"Training set accuracy: {KNN.score(X_train, y_train)}; Test set accuracy: {KNN.score(X_test, y_test)}"
#%%
#MLP NEURAL NETWORK
'''
use a multi-layered perceptron to predict labels
'''
#NO GPU SUPPORT FOR SKLEARN
from sklearn.neural_network import MLPClassifier

#adam: all-round solver for data
#hidden_layer_sizes: no. of nodes/no. of hidden weights used to obtain final weights;
#match with input features
#alpha: regularization parameter that shrinks weights toward 0 (the greater the stricter)
MLP = MLPClassifier(hidden_layer_sizes = 100, solver='adam', alpha=0.01 )
MLP.fit(X_train, y_train)
y_test = MLP.predict(X_test)
f"Training set accuracy: {MLP.score(X_train, y_train)}; Test set accuracy: {MLP.score(X_test, y_test)}"
#%%
#MEASURE ACCURACY OF PREDICTION
#CREATE CONTAINER FOR ACCURACY METRICS

#plot accuracy
#fig, ax = plt.subplots(2,1, figsize=(15,8))

#The index the dataframe we created up above. Equivalent to [0, 1, ..., 28, 29]
#x = df.index.values
#Column 'a' from df.
#y = df['x']

#ax.plot(x, y)
'''
pick accuracy measures
store the value in a data frame
'''
#%%
#PASS PREDICTED VALUES TO APP
'''
let the app display the suggested value in a separate window and ask for confirmation
"correct" "wrong prediction"
'''
#%%
#AWAIT RESPONSE FROM APP
'''
await user input
'''
#%%
#ADJUST WEIGHTS AND APPEND TO CSV FILE FOR FUTURE TRAINING
'''
allocate more weight to correct predictions for later training
create a new column for the weights as numerical values or with a marker
'''
#%%
#WRITE THIS TRANSACTION IN ITS ENTIRETY TO A CSV ROW THAT IS THE TRAINING DATA
'''
write it on a per-line basis to the csv that will sit in the flask folder and will later be available for training
'''
with open('Proportions-Auto recovered.csv','a') as newFile:
    newFileWriter=csv.writer(newFile)
    newFileWriter.writerow([X1, X2, X3, X4, X5])
    return(X2, X3, X4, X5)
    print("row has been written to training data...")
    break

#ALTERNATIVE VERSION
#import pandas as pd
##WILL BE SAVED IN THE SAME FOLDER AS THIS SCRIPT
# Create a Pandas Excel writer using XlsxWriter as the engine.
#def export_df():
#    sheet_name = 'Sheet_1'
#    writer     = pd.ExcelWriter('filename.xlsx', engine='xlsxwriter')
#    df.to_excel(writer, sheet_name=sheet_name)

# Access the XlsxWriter workbook and worksheet objects from the dataframe.
#    workbook  = writer.book
#    worksheet = writer.sheets[sheet_name]

# Adjust the width of the first column to make the date values clearer.
#    worksheet.set_column('A:A', 20)

# Close the Pandas Excel writer and output the Excel file.
#    writer.save()