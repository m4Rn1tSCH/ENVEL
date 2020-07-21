# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:14:32 2020

@author: bill-
"""

"""
Preparations script to apply prediction to the dataframe row by row"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Python_df_encoder_user import df_encoder
from Python_df_encoder_user import split_data
from Python_classifier_testing import pipeline_xgb
from Python_store_pickle import store_pickle
from Python_open_pickle import open_pickle
from Python_pull_data import pull_df

# use these columns as features
# dropped amount_mean_lag7 to avoid errors
feat_merch = ['description', 'transaction_category_name', 'amount', 'state',
              'city', 'transaction_base_type', 'transaction_origin']
# pull data and encode
df = df_encoder(rng=14,
                spending_report=False,
                plots=False,
                include_lag_features=False)
# just pull raw data
df_enc = pull_df(rng=14,
                 spending_report=False,
                 plots=False)

# create the dict for encoded feature
encoding_features = ['primary_merchant_name']
UNKNOWN_TOKEN = '<unknown>'
embedding_maps = {}
for feature in encoding_features:
    unique_list = df_enc[feature].unique().astype('str').tolist()
    unique_list.append(UNKNOWN_TOKEN)
    le = LabelEncoder()
    le.fit_transform(unique_list)
    embedding_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))


X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, \
    X_test_minmax, y_train, y_test = split_data(df= df,
                                                features = feat_merch,
                                                test_size=0.2,
                                                label='primary_merchant_name')

# convert train data to ndarray to avoid feature_names mismatch error
X_array = X_train.values
y_array = y_train.values
Xt_array = X_test.values
yt_array = y_test.values
# X_train and y_train used to train pipeline
xgb_clf_object = pipeline_xgb(x=X_array,
                              y=y_array,
                              test_features=Xt_array,
                              test_target=yt_array,
                              verb=True)

y_pred = xgb_clf_object.predict(Xt_array)
for i in y_pred:
    print(i)
    print(list(embedding_maps.keys())[list(embedding_maps.values()).index(i)])
# generator statement; generated once and exhausted
#dec = [embedding_maps[k] for k in sorted(embedding_maps.keys())]
# merch_val = [v for v in embedding_maps.values()]
# test = {key:value for key, value in embedding_maps.items()}

# iterate through rows; apply the xgb model to each set of feat
# df needs to boast the same columns as the training data
# BUG - still prints whole dict


#try:
    #for company, value in embedding_maps.items():
        # for index, row in df[feat_merch].iterrows():
        #     # prediction with model object per row
        #     y_pred = xgb_clf_object.predict(row)


            # print(list(embedding_maps.keys())[list(embedding_maps.values()).index(y_pred)])

        # for (i,company, value) for i in y_pred and for comapny, value in embedding_maps.items():
        #     if i == value:
        #         print("Prediction: ", i, "Company: ", company)

            # if value in y_pred:
            #     print("Index: ", index, "Prediction: ", y_pred, "Company: ", company)

# except:
#     print("Encoding not found: ", y_pred, "appending to error list")
#     new_encodings_list.append(y_pred)
#     pass



    # insert query into dataframe (PROBLEM FOR-LOOP in SQL)
#%%
'''
Catching the predictions and converting them back to merchants
Should the prediction turn out to be wrong ask for input by the user
Label needs to be primary_merchant_name
'''
# def merch_pred():


#     dec = [embedding_maps[k] for k in sorted(embedding_maps.keys())]
#     merch_val = [v for v in embedding_maps.values()]

#     test = {key:value for key, value in embedding_maps.items()}
# merch_list = []
# for i in dec:
#     if y_pred == value:
#         print(value)
#         merch_list.append(value)

#     return merch_list

#%%
# store trained model as pickle
store_pickle(model=xgb_clf_object)

# open the model; located in the current folder
trained_model = open_pickle(model_file="gridsearch_model.sav")
#%%
# BUGGED
# generate dict with merchants
# def feature_dict(feature):

#     data = df_encoder(rng=14,
#                 spending_report=False,
#                 plots=False,
#                 include_lag_features=False)
#     # take feature and convert it to a dictionary
#     feature = 'primary_merchant_name'
#     unique_list = data[feature].unique().astype('str').tolist()
#     UNKNOWN_TOKEN = "<unknown>"
#     unique_list.append(UNKNOWN_TOKEN)
#     LabelEncoder().fit_transform(unique_list)

#     # dict with original unique permutations as keys and transformed as values
#     val_dict = {}
#     val_dict = dict(zip(LabelEncoder().fit_transform(unique_list), LabelEncoder().inverse_transform())
#     yield val_dict

