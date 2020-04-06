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
'''

#establish a connection to the Yodlee DB
import psycopg2
from psycopg2 import OperationalError

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
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection


#insert a value into the DB
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


#delete comments
delete_comment = "DELETE FROM comments WHERE id = 2"
execute_query(connection, delete_comment)


#add this part at the end to make the module executable as script
#takes arguments here (self)
#
    if __name__ == "__main__":
        import sys
        spending_report(int(sys.argv[1]))