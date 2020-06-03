# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:01:18 2020

@author: bill-
"""
from flask import Flask
from Python_TF2_NN_regression import nn_regression

app = Flask(__name__)

@app.route('/nn_reg')
@nn_tf
def nn_pipeline(self):
    return 'Neural network pipeline has finished'