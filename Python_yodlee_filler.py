#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:08:29 2020

@author: bill
"""

'''Module to fill the yodlee data set

1.) Primary_Merchant_Name
2.) City
If the gaps are filled, the full set is available to train classifiers, regressors
and neural network
'''

# pull state data + encode
from Python_df_encoder_state import df_encoder_state as enc

df = enc(state='NH',
         spending_report=False,
         plots=False,
         include_lag_features=False)
# prepare variables for prediciton workflow
# dropped: panel_file_created_date, user_score, account_score
xgb_feat_merch = ['description', 'transaction_category_name', 'amount', 'state',
                  'city', 'transaction_base_type', 'transaction_origin',
                  'amount_mean_lag7']
X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax,\
y_train, y_test = split_data(df=df,
                             features=xgb_feat_merch,
                             test_size=0.2,
                             label='primary_merchant_name')

# run classifier (top models: xgbclf + svc)
import xgbc_class

pipeline_xgb()

import Python_store_pickle
store_pickle(model'xgb_model')
# insert prediction result per row in the missing field
# for row in df df.iloc[row]
# iteration + label encoder dict

# pass to insert value statement that will be sent to SQL db