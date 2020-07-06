# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:53:49 2020

@author: bill-
"""
"""
Resource file directly called by the flask restful API
This part delivers with imports the classes and functions to the API main module
workflow is integrated inside the class and can contain several functions
    data is the data passed as argument
    id is the id passed as argument
"""
from flask_restful import Resource
from flask import request

# import functions from other folders with resources to unify them in this class
'''from folder.folder_1.folder_2 import function
example folder: .../ml_code/model_data/yodlee_encoder.py with function called df_encoder'''
# import all neccesary ml functions
from ml_code.model_data.yodlee_encoder import df_encoder

# create the flow for data input and running it through the ml functions/models
# this class is added as a resource to the main body
class Foo(Resource):
    def get(self, id):
        data = request.form['data']
        return {'data': data, 'id': id}