# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:51:40 2020

@author: bill-
"""

'''
This script contains all SQL components
-make a connection to the Yodlee DB
-insert records to it
-delete records

when it throws error about transaction blocked enter: rollback
this reverts old incorrect queries

THE SCRIPT ONLY CONDUCTS QUERIES; THE OUTPUT IS A TUPLE!
THE MODULE IT IS CONNECTED TO WILL CONVERT THE PULLED CONTENT TO A DF
'''

#establish a connection to the Yodlee DB
import psycopg2
from psycopg2 import OperationalError
from psycopg2 import pool
#import PostgreSQL_access
#%%
name = "postgres"
user = "envel_yodlee"
pw = "Bl0w@F1sh321"
host = "envel-yodlee-datasource.c11nj3dc7pn5.us-east-2.rds.amazonaws.com"
port = "5432"
#create_connection(name, user, pw, host, port)
#%%
#assign connection object as variable + use in further functions
def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print(f"Connection to PostgreSQL {db_name} successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection
#%%
def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except OperationalError as e:
        print(f"The error '{e}' occurred")
#%%
#example query for transaction in MA
#select_users = "SELECT * FROM bank_record WHERE state = 'MA'"
#generates a tuple output
#transaction_query = execute_read_query(connection, select_users)
#%%
    #insert a value into the DB
def insert_val():
    create_users = """
    INSERT INTO
      users (name, age, gender, nationality)
    VALUES
      ('James', 25, 'male', 'USA'),
      ('Leila', 32, 'female', 'France'),
      ('Brigitte', 35, 'female', 'England'),
      ('Mike', 40, 'male', 'Denmark'),
      ('Elizabeth', 21, 'female', 'Canada');
    """
    return 'edit_msg'
    ############################### alternative version
    execute_query(connection, create_users)

    users = [
        ("James", 25, "male", "USA"),
        ("Leila", 32, "female", "France"),
        ("Brigitte", 35, "female", "England"),
        ("Mike", 40, "male", "Denmark"),
        ("Elizabeth", 21, "female", "Canada"),
    ]

    user_records = ", ".join(["%s"] * len(users))

    insert_query = (
        f"INSERT INTO users (name, age, gender, nationality) VALUES {user_records}"
    )

    connection.autocommit = True
    cursor = connection.cursor()
    cursor.execute(insert_query, users)
    return 'edit_msg'
#%%
def delete_val():
    #delete comments
    delete_comment = "DELETE FROM comments WHERE id = 2"
    execute_query(connection, delete_comment)
    return 'values deleted'

    #add this part at the end to make the module executable as script
    #takes arguments here (self)
#%%
'''
IMPORTANT: This closes all connections even those that are in use by applications!
    Use with caution!
'''
#close a single connection pool
def close_connection():
    pool.SimpleConnectionPool.closeall


    if __name__ == "__main__":
        import sys
        close_connection(int(sys.argv[1]))