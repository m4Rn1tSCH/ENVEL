#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:13:06 2020

@author: bill
"""
import pickle

#flask connection in respective pipeline folder
def open_pickle(model_file):

    """
        Usage of a Pickle Model -Loading of a Pickle File

    model file can be opened either with FILE NAME
    open_pickle(model_file="gridsearch_model.sav")
    INTERNAL PARAMETER
    open_pickle(model_file=model_file)
    """

    with open(model_file, mode='rb') as m_f:
        grid_search = pickle.load(m_f)
        result = grid_search.score(X_test, y_test)
        print("Employed Estimator:", grid_search.get_params)
        print("--------------------")
        print("BEST PARAMETER COMBINATION:", grid_search.best_params_)
        print("Training Accuracy Result: %.4f" %(result))
        return 'grid_search parameters loaded'