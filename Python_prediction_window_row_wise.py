# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:14:32 2020

@author: bill-
"""

"""
Preparations script to apply prediction to the dataframe row by row"""


from Python_df_encoder_user import df_encoder
from Python_df_encoder_user import split_data
from Python_Classifier_Testing import pipeline_xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from Python_store_pickle import store_pickle
from Python_open_pickle import open_pickle
from Python_SQL_connection import insert_val

# use these columns as features
# dropped amount_mean_lag7 to avoid errors
feat_merch = ['description', 'transaction_category_name', 'amount', 'state',
              'city', 'transaction_base_type', 'transaction_origin']

df = df_encoder(rng=14,
                spending_report=False,
                plots=False,
                include_lag_features=False)

X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, \
    X_test_minmax, y_train, y_test = split_data(df= df,
                                                features = feat_merch,
                                                test_size=0.2,
                                                label='primary_merchant_name')
# X_train and y_train used to train pipeline
xgb_clf_object = pipeline_xgb()

# store trained model as pickle
store_pickle(model=xgb_clf_object)

# iterate through rows; apply the xgb model to each set of feat
for index, row in df.iterrows():
    print(row[feat_merch])
    # prediction with model object per row
    y_pred = xgb_clf_object.predict(row[feat_merch])
    # insert query into dataframe (PROBLEM FOR-LOOP in SQL)