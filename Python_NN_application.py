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
import matplotlib.pyplot as plt
import seaborn
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#custom packages
#import Python_df_label_encoding

#%%
#converting the link
link = r"C:\Users\bill-\Desktop\Q_test_data_v2.csv"
link_1 = link.replace(os.sep, '/')
file = ''.join(link_1)
'''
This test data needs more rows than features that have been selected or various
functions might fail
'''
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
#for jupyter-notebook; not practical here
#seaborn.pairplot(df)
#seaborn.heatmap(df)
#seaborn.clustermap(df)
#%%
'''
                APPLICATION OF LABELENCODER####
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
#predicting STUDENT with all other columns as features EXCEPT STUDENT
#drop the students column which is the target to avoid overfitting
#all remaining columns will be the features
feature_df = df.copy()
model_features = feature_df.drop('Student', axis = 1)
model_label = feature_df['Student']

X_train, X_test, y_train, y_test = train_test_split(model_features, model_label,
                                                    shuffle = True,
                                                    test_size = 0.3)
#create a validation set from the  training set


print(f"Shape of the split training data set: X_train:{X_train.shape}")
print(f"Shape of the split training data set: X_test: {X_test.shape}")
print(f"Shape of the split training data set: y_train: {y_train.shape}")
print(f"Shape of the split training data set: y_test: {y_test.shape}")
#%%
#select statisitically significant features with Age as target variable
#chi2 for non-negative ONLY!!
#other score_functions : f_classif; f_regression; mutual_info_regressiom
#all features but one that becomes the label
features = len(df.columns) - 10
k_best = SelectKBest(score_func = f_classif, k = 'all')
k_best.fit(df, df['Age'])
#optimal parameters picked
k_best.scores_
k_best.pvalues_
#%%
#select statisitically significant features with Student as target variable
#deduct 10 features from all available ones (20 left here)
#only the 20 best ones with the strongest correlation will be picked
features = len(df.columns) - 10
k_best = SelectKBest(score_func = f_regression, k = features)
k_best.fit(df, df['Student'])
#optimal parameters picked
feat_list = []
for x in k_best.pvalues_:
    if x <= 0.05:
        print(x)
        feat_list.append(x)
    else:
        pass
#print(k_best.scores_)
#print(k_best.pvalues_)

#%%
#############APPLICATION OF RECURSIVE FEATURE ELIMINATION/LOGISTIC REGRESSION###########################
#Creating training and testing data
train = df.sample(frac = 0.5, random_state = 200)
test = df.drop(train.index)

#pick feature columns to predict the label
#y_train/test is the target label that is to be predicted
#PICKED LABEL = FICO numeric
cols = ["type", "amount", "isCredit", "returnCode", "feeCode", "subTypeCode", "subType", "check", "Student", "account_balance", "Age", "CS_FICO_str", "CS_internal"]
X_train = train[cols]
y_train = train['CS_FICO_num']
X_test = test[cols]
y_test = test['CS_FICO_num']
#build a logistic regression and use recursive feature elimination to exclude trivial features
log_reg = LogisticRegression()
# create the RFE model and select the eight most striking attributes
rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
rfe = rfe.fit(X_train, y_train)
#selected attributes
print('Selected features: %s' % list(X_train.columns[rfe.support_]))
print(rfe.ranking_)

#Use the Cross-Validation function of the RFE modul
#accuracy describes the number of correct classifications
rfecv = RFECV(estimator = LogisticRegression(), step = 1, cv = 8, scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

#plot number of features VS. cross-validation scores
plt.figure(figsize = (10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
#%%
#############APPLICATION OF A RANDOM FOREST REGRESSOR##################
RFR = RandomForestRegressor()
#set up the parameters as a dictionary
parameters = {'n_estimators': [5, 10, 100],
              #'criterion': ['mse'],
              #'max_depth': [5, 10, 15],
              #'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5]
             }

#set up the GridCV
#cv determines the cross-validation splitting strategy /to specify the number of folds in a (stratified)KFold
gridcv = GridSearchCV(estimator = RFR, param_grid = parameters,
                        cv = 5,
                        n_jobs = -1,
                        verbose = 1)

grid = gridcv.fit(X_train, y_train)

#activate the best combination of parameters
RFR = grid.best_estimator_
RFR.fit(X_train, y_train)
#%%
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

y_pred = RFR.predict(X_test)

#in case of rescaling, activate these lines:
#y_pred = y_pred * (max_train - min_train) + min_train
#y_validation_RF = y_validation * (max_train - min_train) + min_train

#if not, keep this one:
y_validation_RF = y_test

print('R2 score = ', r2_score(y_validation_RF, y_pred), '/ 1.0')
print('MSE score = ', mean_squared_error(y_validation_RF, y_pred), '/ 0.0')
#%%
###########################################################
'''
                        APPLICATION OF KERAS
'''
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
#%%
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

#building the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(1))

#compiling the model
model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) #mse: mean_square_error
model.summary()
#%%

#training of the model
epochs_total = 1000
epochs_step = 250
epochs_ratio = int(epochs_total / epochs_step)
hist = np.array([])
#######
#building the epochs
for i in range(epochs_ratio):
    history = model.fit(features, targets, epochs=epochs_step, batch_size=100, verbose=0)

    #evaluating the model on the training and testing set
    print("Step : " , i * epochs_step, "/", epochs_total)
    score = model.evaluate(features, targets)
    print("Training MSE:", score[1])
    score = model.evaluate(features_validation, targets_validation)
    print("Validation MSE:", score[1], "\n")
    hist = np.concatenate((hist, np.array(history.history['mean_squared_error'])), axis = 0)
#%%

#plot metrics
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
'''
                APPLICATION OF SKLEARN NEURAL NETWORK
'''

#NEURAL NETWORK
#NO GPU SUPPORT FOR SKLEARN
from sklearn.neural_network import MLPClassifier

#adam: all-round solver for data
#hidden_layer_sizes: no. of nodes/no. of hidden weights used to obtain final weights;
#match with input features
#alpha: regularization parameter that shrinks weights toward 0 (the greater the stricter)
MLP = MLPClassifier(hidden_layer_sizes = 1000, solver='adam', alpha=0.001 )
MLP.fit(X_train, y_train)
y_val = MLP.predict(X_test)
#y_val.reshape(-1, 1)
print(f"Training set accuracy: {MLP.score(X_train, y_train)}; Test set accuracy: {MLP.score(X_test, y_test)}")
#%%
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#y_val = y_pred as the split is still unfisnished
print('R2 score = ', r2_score(y_val, y_pred), '/ 1.0')
print('MSE score = ', mean_squared_error(y_val, y_pred), '/ 0.0')
#%%
'''
                    APPLICATION OF PYTORCH
'''
#GENERAL
    #root package
    #import torch
    #dataset representation and loading
    #from torch.utils.data import Dataset, Dataloader
    #set up x and y and use the non-processed data for analysis

#NEURAL NETWORK API
    #computation graph
    #import torch.autograd as autograd
    #tensor node in the computation graph
    #from torch import Tensor
    #neural networks
    #import torch.nn as nn
    #layers, activations and more
    #import torch.nn.functional as F
    #optimizers e.g. gradient descent, ADAM, etc.
    #import torch.optim as optim
    #hybrid frontend decorator and tracing jit
    #from torch.jit import script, trace


#define the network
    #import torch
    #import torch.nn as nn
    #import torch.nn.functional as F
#%%
#required packages
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
#load the data set
dataset = pd.read_csv(r'E:Datasets\customer_data.csv')

#exploratory data analyis
dataset.shape
dataset.head()
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
dataset.Exited.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=['skyblue', 'orange'], explode=(0.05, 0.05))
sns.countplot(x='Geography', data=dataset)
sns.countplot(x='Exited', hue='Geography', data=dataset)
#%%
#Preprocessing of data
#conversion of columns + preparation of cols for NN
dataset.columns
#categorical columns for boolean info or encoded strings
categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
#numerical columns for integer or float values
numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
#the target value or the label that is to be predicted
outputs = ['Exited']

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')


dataset.dtypes
dataset['Geography'].cat.categories
dataset['Geography'].head().cat.codes

#encode of categorical values to numerical values
geo = dataset['Geography'].cat.codes.values
gen = dataset['Gender'].cat.codes.values
hcc = dataset['HasCrCard'].cat.codes.values
iam = dataset['IsActiveMember'].cat.codes.values
#stack them as an N-dimensional vector to feed it into the Pytorch network
categorical_data = np.stack([geo, gen, hcc, iam], 1)
#print the first 10 values
categorical_data[:10]
#convert the categorical data to a tensor ready for Pytorch
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
categorical_data[:10]
#convert the numerical data to a tensor ready for Pytorch
numerical_data = np.stack([dataset[col].values for col in numerical_columns], 1)
numerical_data = torch.tensor(numerical_data, dtype=torch.float)
numerical_data[:5]
#preparation of labels
outputs = torch.tensor(dataset[outputs].values).flatten()
outputs[:5]

print(categorical_data.shape)
print(numerical_data.shape)
print(outputs.shape)
#%%
#conversion to tensors to feed into Neural Network model
categorical_column_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]
print(categorical_embedding_sizes)

total_records = 10000
test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]
numerical_train_data = numerical_data[:total_records-test_records]
numerical_test_data = numerical_data[total_records-test_records:total_records]
train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]

print(len(categorical_train_data))
print(len(numerical_train_data))
print(len(train_outputs))

print(len(categorical_test_data))
print(len(numerical_test_data))
print(len(test_outputs))

#building the model and passing training  data with labels
#set up everything as a class to summarize all steps
#model inherits from Pytorch's NN.module class
class Model(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x

#initializing the model
#in square brackets; structure of hidden layers, 200, 100 and 50 neurons respectively
pytorch_model = Model(categorical_embedding_sizes, numerical_data.shape[1], 2, [200,100,50], p=0.4)
print(pytorch_model)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300
aggregated_losses = []

for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data, numerical_train_data)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

plt.plot(range(epochs), aggregated_losses)
plt.ylabel('Loss')
plt.xlabel('epoch');

#using a trained model for predictions
with torch.no_grad():
    y_val = model(categorical_test_data, numerical_test_data)
    loss = loss_function(y_val, test_outputs)
print(f'Loss: {loss:.8f}')

print(y_val[:5])

#finding out the maximum of predictions
y_val = np.argmax(y_val, axis=1)
print(y_val[:5])

#examine accuracy of the predictions
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(test_outputs,y_val))
print(classification_report(test_outputs,y_val))
print(accuracy_score(test_outputs, y_val))

#%%
##DEFINING THE NET OF LAYERS

##STRUCTURE OF A LAYER
#1 input image channel, 6 output channels, 3x3 square convolution
#conv1 = nn.Conv2d(1, 6, 3)
##FORWARD OPERATION
#one forward operation with relu as activation
#class Net(nn.Module):

#    def __init__(self):
#        super(Net, self).__init__()
#        # 1 input image channel, 6 output channels, 3x3 square convolution
#        # kernel
#        self.conv1 = nn.Conv2d(1, 6, 3)
#        self.conv2 = nn.Conv2d(6, 16, 3)
#        # an affine operation: y = Wx + b
#        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 10)

#    def forward(self, x):
#        # Max pooling over a (2, 2) window
#        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#        # If the size is a square you can only specify a single number
#        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#        x = x.view(-1, self.num_flat_features(x))
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x

#    def num_flat_features(self, x):
#        size = x.size()[1:]  # all dimensions except the batch dimension
#        num_features = 1
#        for s in size:
#            num_features *= s
#        return num_features

#net = Net()
#print(net)
#%%
#print learnable parameters
#params = list(net.parameters())
#print(len(params))
#print(params[0].size())  # conv1's .weight
#%%

#CREATION OF TENSORS
#tensor with independent N(0,1) entries
#torch.randn(*size)
#tensor with all 1's [or 0's]
#torch.[ones|zeros](*size)
#create tensor from [nested] list or ndarray L
#torch.Tensor(L)
#clone of x
#x.clone()
#code wrap that stops autograd from tracking tensor history
#with torch.no_grad():
#arg, when set to True, tracks computation
#    requires_grad=True
#history for future derivative calculations

##building the model
#setting up layers
#model = Sequential(
#    torch.layers())
#%%
##############################################################
'''
                    APPLICATION OF TENSORFLOW
'''
#future needs to be run first
#eager execution needs to be run right after the TF instantiation to avoid errors
from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column
from tensorflow.keras import layers
#%%
#custom train test split
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
#%%
'''
V_1COLUMN STRUCTURE
This is an overview about the choice of columns for the model and how to preprocess them before
compiling the model and training it

Data columns (total 30 columns):
transactionCount       6 non-null int64
transactionId          6 non-null int64
masterId               6 non-null int64
customerId             6 non-null int64
type                   6 non-null object categorical (wrapped in embedding)x
typeCode               6 non-null int64 x
tag                    6 non-null int64
friendlyDescription    6 non-null int64 x
description            6 non-null int64
status                 6 non-null int64
createdDate            6 non-null object bucketized? x
amount                 6 non-null float64 x
isCredit               6 non-null object categorical (wrapped in embedding) x
settledDate            6 non-null object
availableDate          6 non-null object
voidedDate             6 non-null object
returnCode             6 non-null int64 categorical
feeCode                6 non-null int64 categorical (wrapped in embedding) x
feeDescription         6 non-null object
cardId                 6 non-null int64
subTypeCode            6 non-null object
subType                6 non-null object
institutionName        6 non-null object categorical (wrapped in indicator) x
check                  6 non-null object categorical (wrapped in embedding) x
Student                6 non-null int64 LABEL
account_balance        6 non-null int64 bucketized (wrapped in embedding) x
Age                    6 non-null int64 bucketized x
CS_internal            6 non-null int64 bucketized/ categorical x
CS_FICO_num            6 non-null int64 categorical/bucketized if as strings (wrapped in embedding) x
CS_FICO_str            6 non-null int64 categorical

V_2 COLUMN STRUCTURE
Data columns (total 30 columns):
transactionCount       5452 non-null int64
transactionId          5452 non-null int64
masterId               5452 non-null float64
customerId             5452 non-null int64
type                   5452 non-null object x
typeCode               5452 non-null int64 numeric x
tag                    5452 non-null int64
friendlyDescription    5452 non-null object cat with hashed feat (wrapped in embedding)
description            5452 non-null float64
status                 5452 non-null object
createdDate            5452 non-null object
amount                 5452 non-null float64 numeric x
isCredit               5452 non-null object categorical (wrapped in embedding)
settledDate            5452 non-null object
availableDate          5452 non-null object
voidedDate             5452 non-null object
returnCode             5452 non-null int64
feeCode                5452 non-null object
feeDescription         5452 non-null object
cardId                 5452 non-null int64
subTypeCode            5452 non-null object
subType                5452 non-null object
institutionName        5452 non-null object categorical (wrapped in indicator)
check                  5452 non-null object categorical (wrapped in embedding)
Student                5452 non-null int64 LABEL
account_balance        5452 non-null object
Age                    5452 non-null int64 bucketized x
CS_internal            5452 non-null int64 bucketized/ categorical x
CS_FICO_num            5452 non-null int64 categorical/bucketized if as strings (wrapped in embedding) x
CS_FICO_str            5452 non-null int64 categorical
'''
#%%
#create tuples for lists to organize the columns conversion
##Tuple (or list) for the bank list to refer to the length
#banks = list('Bank of America','Toronto Dominion Bank', 'Citizens Bank', 'Webster Bank',
#      'CHASE Bank', 'Citigroup', 'Capital One', 'HSBC Bank USA',
#      'State Street Corporation','MUFG Union Bank', 'Wells Fargo & Co.', 'Barclays',
#      'New York Community Bank', 'CIT Group', 'Santander Bank',
#      'Royal Bank of Scotland', 'First Rand Bank', 'Budapest Bank')

#trans_type = list('CorePro Deposit', 'CorePro Withdrawal', 'Internal CorePro Transfer',
#                   'Interest Paid', 'CorePro Recurring Withdrawal',
#                   'Manual Adjustment', 'Interest Adjustment')

##STEP 1
'''
attempt 12/12/ ; 48%-51% accuracy to predict student with features: TYPECODE + AMOUNT + RETURNCODE + CS_FICO_NUM + AGE + CROSSED(CS_FICO;AGE)
attempt_2 12/12/ ; 49%-50% accuracy to predict student with features: TYPECODE + FEE_CODE + AMOUNT + RETURNCODE + INSTITUTION_NAMES + CS_FICO_NUM + AGE + CROSSED(CS_FICO;AGE)
'''
#feature columns to use in the layers
feature_columns_container = []

#numeric column needed in the model
#wrap all non-numerical columns with indicator col or embedding col
######IN V2 STATUS IS NUMERICAL; THATS WHY IT WILL THROW "CAST STRING TO FLOAT IS NOT SUPPORTED" ERROR######
#the argument default_value governs out of vocabulary values and how to replace them

for header in ['typeCode', 'amount', 'returnCode', 'CS_FICO_num']:
  feature_columns_container.append(feature_column.numeric_column(header))

#bucketized column

#categorical column with vocabulary list
#type_col = feature_column.categorical_column_with_vocabulary_list(
#        'type', ['CorePro Deposit',
#                 'CorePro Withdrawal',
#                 'Internal CorePro Transfer',
#                 'Interest Paid',
#                 'CorePro Recurring Withdrawal',
#                 'Manual Adjustment',
#                 'Interest Adjustment'])
#type_pos = feature_column.indicator_column(type_col)
#type_pos_2 = feature_column.embedding_column(type_col, dimension = 8)
#feature_columns_container.append(type_pos)
#feature_columns_container.append(type_pos_2)

#idea: words or fragments are a bucket and can be used to recognize recurring bills
#friendly_desc = feature_column.categorical_column_with_hash_bucket(
#        'friendlyDescription', hash_bucket_size = 2500)
#fr_desc_pos = feature_column.embedding_column(friendly_desc, dimension = 250)
#feature_columns_container.append(fr_desc_pos)

#created_date = feature_column.categorical_column_with_hash_bucket(
#        'createdDate', hash_bucket_size = 365)
#set the indicator column
#cr_d_pos = feature_column.indicator_column(created_date)
#feature_columns_container.append(cr_d_pos)

#entry = feature_column.categorical_column_with_vocabulary_list(
#        'isCredit', ['Y', 'N'])
#set the embedding column
#entry_pos = feature_column.embedding_column(entry, dimension = 3)
#feature_columns_container.append(entry_pos)

#ret_c = feature_column.categorical_column_with_vocabulary_list(
#        'returnCode', ['RGD', 'RTN', 'NSF'])

fee = feature_column.categorical_column_with_vocabulary_list('feeCode',
                                                             ['RGD',
                                                              'RTN',
                                                              'NSF'])
#set the embedding column
fee_pos = feature_column.embedding_column(fee, dimension = 3)
feature_columns_container.append(fee_pos)

#check = feature_column.categorical_column_with_vocabulary_list('check',
#                                                               ['Y', 'N'])
#set the indicator column
#check_pos = feature_column.embedding_column(check, dimension = 2)
#feature_columns_container.append(check_pos)

#acc_bal = feature_column.categorical_column_with_vocabulary_list('account_balance',
#                                                                 ['u100',
#                                                                  'o100u1000',
#                                                                  'o1000u10000',
#                                                                  'o10000'])
#set the indicator value
#acc_bal_pos = feature_column.embedding_column(acc_bal, dimension = 10)
#feature_columns_container.append(acc_bal_pos)


age = feature_column.bucketized_column(feature_column.numeric_column('Age'),
                                       boundaries = [18, 20, 22, 26, 31, 35])
feature_columns_container.append(age)


#cs_internal = feature_column.categorical_column_with_vocabulary_list('CS_internal',
#                                                                       ['Poor',
#                                                                        'Average',
#                                                                        'Excellent'])
#set the indicator value
#cs_positive = feature_column.embedding_column(cs_internal)
#feature_columns_container.append(cs_positive)

#FICO 700 is the initial score and also the average in the US
#The CS_FICO_num column is in this version converted to a bucketized column
#instead of passing it to the feature_column_container
#columns remains bucketized without wrapping to embedded or indicator
fico_num = feature_column.bucketized_column(feature_column.numeric_column('CS_FICO_num'),
                                                boundaries = [300,
                                                              580,
                                                              670,
                                                              700,
                                                              740,
                                                              800,
                                                              850])
feature_columns_container.append(fico_num)


institutions = feature_column.categorical_column_with_vocabulary_list(
        'institutionName', [
            'Bank of America', 'Toronto Dominion Bank', 'Citizens Bank', 'Webster Bank',
            'CHASE Bank', 'Citigroup', 'Capital One', 'HSBC Bank USA',
            'State Street Corporation', 'MUFG Union Bank', 'Wells Fargo & Co.', 'Barclays',
            'New York Community Bank', 'CIT Group', 'Santander Bank',
            'Royal Bank of Scotland', 'First Rand Bank', 'Budapest Bank'
            ])
institutions_pos = feature_column.indicator_column(institutions)
feature_columns_container.append(institutions_pos)

crossed_feat = feature_column.crossed_column([age, fico_num], hash_bucket_size = 1000)
crossed_feat = feature_column.indicator_column(crossed_feat)
feature_columns_container.append(crossed_feat)

###########EXAMPLES#######
#numeric column
#age = feature_column.numeric_column("age")

#categorical column with vocabulary list
#thal = feature_column.categorical_column_with_vocabulary_list(
#      'thal', ['fixed', 'normal', 'reversible'])

#bucketized column
#age_buckets = feature_column.bucketized_column(
#   age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

#embedding column
#feature_column.embedding_column(thal, dimension=8)
#feature_columns_container.append(age_buckets)

#hashed feature column
#feature_column.categorical_column_with_hash_bucket(
#      'thal', hash_bucket_size=1000)
#feature_columns_container.append(age_buckets)

#crossed feature column
#feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
#feature_columns_container.append(age_buckets)

#indicator column (like bucketized but with one vital string that is marked a "1")
#also used as a wrapper for categorical columns to ensure wokring feature_layers
##########################
#%%
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle = True, batch_size = 150):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Student')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size = len(dataframe))
  ds = ds.batch(batch_size)
  return ds
######OVERHAUL NEEDED HERE#####
#def make_input_fn(df):
#  def pandas_to_tf(pdcol):
    # convert the pandas column values to float
#    t = tf.constant(pdcol.astype('float32').values)
    # take the column which is of shape (N) and make it (N, 1)
#    return tf.expand_dims(t, -1)

#  def input_fn():
    # create features, columns
#    features = {k: pandas_to_tf(df[k]) for k in FEATURES}
#    labels = tf.constant(df[TARGET].values)
#    return features, labels
#  return input_fn

#def make_feature_cols():
#  input_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
#  return input_columns
##################################
#%%
##STEP 2
#create layers
feature_layer = tf.keras.layers.DenseFeatures(feature_columns_container)
#print(feature_layer)
#%%
#STEP 3
batch_size = 250
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle = True, batch_size = batch_size)
test_ds = df_to_dataset(test, shuffle = True, batch_size = batch_size)
#%%
#STEP 4
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(units = 256, activation = 'relu'),
  layers.Dense(units = 256, activation = 'relu'),
  layers.Dense(units = 256, activation = 'relu'),
  layers.Dense(units = 1, activation = 'sigmoid')
])
#%%
##STEP 5
model.compile(optimizer = 'Adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(train_ds,
          validation_data = val_ds,
          epochs=2)
#%%
##STEP 6
# Check accuracy
loss, accuracy = model.evaluate(test_ds)
print("Accuracy:", accuracy)

##STEP 7
# Infer labels on a batch
predictions = model.predict(test_ds)