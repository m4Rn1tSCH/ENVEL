# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:10:18 2020

@author: bill-
"""
'''
This module sets and activates the flask app in windows
'''
#import subprocess
#subprocess.call(['C:\Users\bill-\OneDrive\Dokumente\Docs Bill\TA_files\functions_scripts_storage\flask_test\flask_console_setup.txt'])
#%%
#run a string in the command shell
#here to set up flask on the machine
import os

def activate_flask():
    script_path = os.path.abspath(r'Python_inc_exp_bal_database_test.py')
    os.system(f"env FLASK_APP={script_path} flask run")
    return 'environment variable set; Flask is being executed...'

def activate_flask_2():
    script_path = os.path.abspath(r'Python_EDA_users_test.py')
    os.system(f"env FLASK_APP={script_path} flask run")
    return 'environment variable set; Flask is being executed...'