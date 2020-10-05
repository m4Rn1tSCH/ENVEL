#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:42:51 2020

@author: bill
"""

import logging
import os

import pandas as pd

import psycopg2
from sqlalchemy import create_engine


logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.ERROR)
logger = logging.getLogger(__name__)


class Connect(object):

    def __init__(self, name=None, database=None):
        self.name = name.lower()
        self.database = database
        self.connection = None
        self.sql = None

    def __str__(self):
        return f'Connection name: {self.name}'


class Postgres(Connect):
    """
    Client to interface with Data Science's various tables and views
    """

    def __init__(self, database=None):
        Connect.__init__(self, name='psql')

        self.engine = None
        self.disposition = 'query'
        self.write_disposition = 'replace'
        self.locals = None

        if database is None:
            self.database   = os.environ.get('ENVEL_POSTGRES_DB')
            self.host       = os.environ.get('ENVEL_POSTGRES_HOST')
            self.user       = os.environ.get('ENVEL_POSTGRES_USERNAME')
            self.password   = os.environ.get('ENVEL_POSTGRES_PASSWORD')
            self.port       = os.environ.get('ENVEL_POSTGRES_PORT')

    def connect(self):

        try:
            self.connection = psycopg2.connect(host=self.host,
                                               user=self.user,
                                               password=self.password,
                                               dbname=self.database,
                                               port=self.port)
            logger.info(f'Successfully connected to {self.database}.')

            return None

        except Exception as e:
            logger.error(f'Connection error: {e}')
            raise

    def execute(self):

        self.connect()
        try:
            with self.connection.cursor() as curr:
                if self.disposition == 'query':
                    curr.execute(self.sql)
                    d = curr.fetchall()
                    c = list(map(lambda x: x[0], curr.description))
                    res = pd.DataFrame(d, columns=c)
                    curr.close()

                    logger.info('Successfully fetched results.')
                    return res

                elif self.disposition in ['update', 'insert', 'delete', 'execute']:
                    curr.execute(self.sql)
                    self.connection.commit()
                    curr.close()

                    logger.info('Successfully altered the DB.')
                    return None

                else:
                    self.connection.commit()
                    curr.close()

                    logger.info('Successfully altered the DB.')
                    return None

        except Exception as e:
            logger.error(f'Query error: {e}')
            raise

        finally:
            self.connection.close()
            logger.info(f'Closed connection to {self.database}.')

    def open_file(self, file_name):
        """Opens file containing SQL text and stores the content in self.sql
        :param file_name: Path to file containing SQL text to be executed
        :type file_name: string
        :return: None - stores SQL text found in file in self.sql
        """
        with open(file_name, 'r') as sqlFile:
            self.sql = sqlFile.read()

        return None

    def query(self, file_name, *args, single_date=None, min_date=None, max_date=None):
        """Establishes a connection to self.database and returns a Pandas dataframe
        containing the data pulled using the query stored in file_name
        :param file_name: Path to file containing SQL text to be executed
        :param single_date: Optional - date parameter limiting data pulled
        :param min_date: Optional - start date parameter limiting data pulled, used with max_date
        :param max_date: Optional - end date parameter limiting data pulled, used with min_date
        :return: Pandas dataframe containing data from self.database, returned by running the query in self.sql
        """

        self.open_file(file_name=file_name)

        if min_date is not None and max_date is not None and single_date is None:
            self.sql = self.sql.format(min_date, max_date)
        elif single_date is not None and min_date is None and max_date is None:
            self.sql = self.sql.format(single_date)
        elif single_date is not None and (min_date is not None or max_date is not None):
            logger.error(
                'Wrong date parameter(s) passed. '
                'Use min_date and max_date to specify a range, or single_date for one date limiter.')
            raise ValueError
        elif args is not None:
            self.sql = self.sql.format() % args
        else:
            logger.info('No date parameter passed. Using sql_text as is.')
            pass

        res = self.execute()

        return res

    def load(self, dataframe, table_name, if_exists=None):
        """Instantiate a SQLAlchemy connection to the Postgres DB and upload the data passed as argument
        :param dataframe: Input data to be uploaded to the Postgres DB
        :type dataframe: Pandas DataFrame
        :param table_name: Input string to identify Postgres DB table to write to
        :type table_name: string
        :param if_exists: Input string to override default (replace) write disposition - options are replace and append
        :type if_exists: string
        :return:
        """
        self.disposition = 'load'
        if if_exists is not None:
            self.write_disposition = if_exists

        self.engine = create_engine(
            f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}',
            use_batch_mode=True)

        dataframe.to_sql(name=table_name,
                         con=self.engine,
                         if_exists=self.write_disposition,
                         index=False,
                         chunksize=100000)

        return dict(
            uri=f'postgres://{self.host}/{self.database}/{table_name}',
            storage_class='cloudsql'
        )

    def update(self, file_name, **kwargs):
        """
        :param file_name:
        :return:
        """
        self.disposition = 'update'
        self.open_file(file_name=file_name)

        if kwargs:
            ls = list()
            values = list()

            for i in range(0, len(kwargs)):
                ls.append('%s')

            for k, v in kwargs.items():
                values.append(v)

            self.sql = self.sql.format(*ls) % tuple(values)

        self.execute()

        return 200, 'OK'

    # TODO: take in table name and delete table
    def delete(self, file_name):
        """
        :param file_name:
        :return:
        """
        self.disposition = 'delete'
        self.open_file(file_name=file_name)

        self.execute()

        return 200, 'OK'

    def exec_script(self, file_name):
        """
        :param file_name:
        :return:
        """
        self.disposition = 'execute'
        self.open_file(file_name=file_name)

        self.execute()

        return 200, 'OK'

    def insert(self, dataframe, table_name):
        """WIP - do not use. This is not ready yet, you will definitely get errors.
        Currently doesn't infer dtypes and includes it in the unnest statement.
        TODO: Need to add a catch to check for dtype, if num then use %({i})d, else %({i})s
        :param dataframe: Input data - dataframe containing data to write to DB
        :type dataframe: Pandas DataFrame
        :param table_name: Value identifying table to write to in Postgres DB
        :type table_name: string
        :return:
        """
        self.disposition = 'insert'

        insert_text = f'INSERT INTO {table_name}('
        select_text = 'SELECT '
        self.locals = {}
        j = 1
        for i in list(dataframe.columns):

            self.locals[f'{i}'] = [r for r in dataframe[i]]

            if j == len(list(dataframe.columns)):
                insert_text = insert_text + f'{i}) '
                select_text = select_text + f' unnest( %({i})s ) '
            else:
                insert_text = insert_text + f'{i}, '
                select_text = select_text + f'unnest( %({i})s ), '
            j += 1

        self.sql = insert_text + select_text

        self.execute()

        return 200, 'OK'