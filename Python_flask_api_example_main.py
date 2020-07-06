# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:43:10 2020

@author: bill-
"""

"""
Documentation for flask and set up of a API
-Declare app first
-wrap this app into an API object
-append decorators leading to classes + corresponding functions with 'add_resource'
-run the function in a local testing environment
"""

from flask import Flask
from flask_cors import CORS
from flask_restful import Api

from resources.foo import Foo

app = Flask(__name__)
CORS(app)
api = Api(app)

# Define api routes with the associated imported class
# resource points to class first, then the class decoratpr and then underlying
# functions of this class (class, class decorator, function that is part of that class)
api.add_resource(Foo, '/Foo', '/Foo/<string:id>')

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True, threaded=True)