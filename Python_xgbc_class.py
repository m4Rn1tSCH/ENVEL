#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:56:48 2020

@author: bill
"""

from xgboost import XGBClassifier

def pipeline_xgb():


    xgbclf = XGBClassifier(verbose=0)
    # Add silent=True to avoid printing out updates with each cycle
    xgbclf.fit(X_train, y_train, verbose=False)

    # make predictions
    y_pred = xgbclf.predict(X_test)
    print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))

    return xgbclf