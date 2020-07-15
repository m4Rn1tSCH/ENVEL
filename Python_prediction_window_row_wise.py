# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:14:32 2020

@author: bill-
"""

"""
Preparations script to apply prediction to the dataframe row by row"""

from Python_df_encoder_user import df_encoder

df = df_encoder(rng=14,
                spending_report=False,
                plots=False,
                include_lag_features=False)

from Python