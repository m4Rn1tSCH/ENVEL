#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:08:29 2020

@author: bill
"""

'''Module to fill the yodlee data set

1.) Primaty_Merchant_Name
2.) City
If the gaps are filled, the full set is available to train classifiers, regressors
and neural network
'''

# pull state data + encode
import Python_df_encoder_state as enc
df = enc(state='ME')
# run classifier (top models: xgbclf + svc)
from envel-machine-learning.ml_code.classification_models.xgbc_class import pipeline_xgb
# insert prediction result per row in the missing field
for row in df df.iloc[row]
# iteration + label encoder dict

# pass to insert value statement that will be sent to SQL db