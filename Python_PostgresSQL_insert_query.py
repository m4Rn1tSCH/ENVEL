#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:01:20 2020

@author: bill
"""

# LOCAL IMPORTS
import psycopg2
from psycopg2 import OperationalError
import numpy as np
import pandas as pd
from ml_code.model_data.raw_data_connection import pull_df
from ml_code.model_data.SQL_connection import create_connection, execute_read_query



db_name = "postgres"
db_user = "envel"
db_pw = "envel"
db_host = "0.0.0.0"
db_port = "5432"

# merch_list = ["DD", "Starbucks", "GAP", "COCA_COLA"]
# test_tuple = tuple(merch_list)
# merch_tuple = [('DD'), ('Starbucks'), ('GAP'), ('COCA_COLA')]

test = ['Tatte Bakery', 'Star Market', 'Stop n Shop', 'Auto Parts Shop',
        'Trader Joes', 'Insomnia Cookies']

def list_insert(values):

    """
    Parameters.
    ------------
    values. List/array/iterable. Values to be inserted into the databank.

    Returns.
    -----------
    'Operation complete' + disconnect
    """


    try:
        connection = create_connection(db_name=db_name,
                                        db_user=db_user,
                                        db_password=db_pw,
                                        db_host=db_host,
                                        db_port=db_port)
        print("-------------")
        cursor = connection.cursor()
        sql_insert_query = """
        INSERT INTO test (test_col_2)
        VALUES (%s);
        """

        # tuple or list works
        for i in values:
        # executemany() to insert multiple rows rows
            cursor.execute(sql_insert_query, (i, ))

        connection.commit()
        print(len(values), "record(s) inserted successfully. \n ------------")

    except (Exception, psycopg2.Error) as error:
        print("Failed inserting record {}".format(error))

    finally:
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("Operation accomplished.\nPostgreSQL connection is closed.")
    print("================")
    return'done'
