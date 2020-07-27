#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:07:54 2020

@author: bill
"""

import pickle
import os

def store_pickle(model):

    """
    stores a pickle model file of a trained algorithm.
    storage in current work directory.
    """

    model_file = "gridsearch_model.sav"
    with open(model_file, mode='wb') as m_f:
        pickle.dump(model, m_f)
    print(f"Model saved in: {os.getcwd()}")
    return model_file