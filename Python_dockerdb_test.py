#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:52:50 2020

@author: bill
"""

# LOCAL IMPORTS
import sys
# sys.path.append('C:/Users/bill-/OneDrive/Dokumente/Docs Bill/TA_files/functions_scripts_storage/envel-machine-learning')
import psycopg2
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

merch_list = ["DD", "Starbucks", "GAP", "COCA_COLA"]
test_tuple = tuple(merch_list)
merch_tuple = [('DD'), ('Starbucks'), ('GAP'), ('COCA_COLA')]
single = ['val_1', 'val_2']


try:
    connection = create_connection(db_name=db_name,
                                    db_user=db_user,
                                    db_password=db_user,
                                    db_host=db_host,
                                    db_port=db_port)

    cursor = connection.cursor()
    sql_insert_query = """
    INSERT INTO test (test_col_2)
    VALUES (%s);
    """
    for i in merch_tuple:
    # executemany() to insert multiple rows rows
        cursor.execute(sql_insert_query, (i, ))
    connection.commit()
    print(len(merch_tuple), "Record inserted successfully.")

except (Exception, psycopg2.Error) as error:
    print("Failed inserting record {}".format(error))

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("Operation accomplished.\nPostgreSQL connection is closed...")

