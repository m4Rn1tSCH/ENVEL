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
from Python_SQL_connection import insert_val, insert_val_alt

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

# array object
y_pred = xgb_clf_object.predict(Xt_array)
#inverse transformation to merchant strings
decoded_merchants = dict(zip(le.classes_, le.inverse_transform(y_pred)))

# insert query into dataframe (PROBLEM FOR-LOOP in SQL)
my_sql_string = """test
                """


# insert values into Yodlee DB
# version 1
#insert_val(query_string) = my_sql_string

# version 2
#insert_val_alt(insertion_val = ,
#               columns = )

# store trained model as pickle
store_pickle(model=xgb_clf_object)

# open the model; located in the current folder
trained_model = open_pickle(model_file="gridsearch_model.sav")
