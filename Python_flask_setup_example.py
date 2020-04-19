# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:01:22 2019

@author: bill-
"""

#how to set up flask on either a built-in server or an external one


########WHAT TO SET UP IN PYTHON
#loading the simplified applications
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'HELLO TATTI! n\ I AM BILLS PC AND I LIKE YOU TOO'


#route tells what URL should trigger the function
#use __main__ only for the actual core of the application
# pick unique names for particular functions if these are imported later
#DONT CALL APPLICATIONS FLASK.PY TO AVOID CONFLICTS WITH FLASK

#RUN THE APPLICATION
#flask command or -m swith in Python

'''
test successful on 3/30/2020
in console when already in the corect/desired folder

produce a virtual environment called venv
only needed to genrate a minial python version and then install flask in it

set the environment variable to the folder where the script is sitting (in Windows)
py -3 -m venv venv
'''

########SETTING THE ENVIRONMENT VARIABLE#######
#$ export FLASK_APP=hello.py
#$ flask run
# * Running on http://127.0.0.1:5000/

'''
#if the python script is set up
#and the module is inside the script
#set the evironment variable to the
'''
####COMMAND PROMPT#####
#the terminal knows what to work with
#C:\path\to\app>set FLASK_APP=hello.py

####for production use##
#local server is set up
#dont use this in production mode
#server is now reachable under this ip (ONLY LOCALLY)
#http://127.0.0.1:5000/
#0.0. will make the PC listen to all public IPs
#$ flask run --host=0.0.0.0
