#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:52:50 2020

@author: bill
"""

# LOCAL IMPORTS
import sys
# sys.path.append('C:/Users/bill-/OneDrive/Dokumente/Docs Bill/TA_files/functions_scripts_storage/envel-machine-learning')

from psycopg2 import OperationalError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_code.model_data.spending_report_csv_function import spending_report as create_spending_report
from ml_code.model_data.raw_data_connection import pull_df
from ml_code.model_data.SQL_connection import insert_val_alt, create_connection, execute_read_query


db_name = "postgres"
db_user = "envel"
db_pw = "envel"
db_host = "0.0.0.0"
db_port = "5432"

connection = create_connection(db_name=db_name,
                                db_user=db_user,
                                db_password=db_user,
                                db_host=db_host,
                                db_port=db_port)


merch = ['merch_1', 'merch_2', 'merch_3']
gm = [i for i in merch]

tuples = [
    ("James", 25, "male", "USA"),
    ("Leila", 32, "female", "France"),
    ("Brigitte", 35, "female", "England"),
    ("Mike", 40, "male", "Denmark"),
    ("Elizabeth", 21, "female", "Canada"),
    ]

insert_val_alt(table='postgres',
               columns='test_col',
               insertion_val=gm)