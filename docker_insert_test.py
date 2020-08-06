#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:01:20 2020

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
from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder

from ml_code.model_data.raw_data_connection import pull_df
from ml_code.model_data.SQL_connection import create_connection, execute_read_query
from ml_code.model_data.split_data_w_features import split_data_feat
from ml_code.classification_models.xgbc_class import pipeline_xgb
from ml_code.classification_models.svc_class import svc_class
from ml_code.model_data.pickle_io import store_pickle, open_pickle


db_name = "postgres"
db_user = "envel"
db_pw = "envel"
db_host = "0.0.0.0"
db_port = "5432"

merch_list = ["DD", "Starbucks", "GAP", "COCA_COLA"]
test_tuple = tuple(merch_list)
merch_tuple = [('DD'), ('Starbucks'), ('GAP'), ('COCA_COLA')]



def docker_test():


    fields = ['unique_mem_id', 'optimized_transaction_date', 'amount', 'description',
              'primary_merchant_name', 'transaction_category_name', 'state',
              'city', 'transaction_base_type', 'transaction_origin']
    
    # TEST DF NOT ENCODED!!
    test_df_1 = pull_df(rng=20,
                      spending_report=False,
                      plots=False)
    test_df_2 = pull_df(rng=22,
                      spending_report=False,
                      plots=False)
    main_df = pd.concat([test_df_1, test_df_2], axis=0)
    # main_df = pull_df(rng=4,
    #                   spending_report=False,
    #                   plots=False)

    for num, user in enumerate(main_df.groupby('unique_mem_id')):
        print(f"User: {user[0]}, {num+1}/10000 users, {round(((num+1)/10000)*100, 2)}%.")


        main_df['optimized_transaction_date'] = pd.to_datetime(
                main_df['optimized_transaction_date'])



        try:
            # Include below if needed unique ID's later:
            # main_df['unique_mem_id'] = main_df['unique_mem_id'].astype(
            #     'str', errors='ignore')
            main_df['amount'] = main_df['amount'].astype('float64')
            main_df['transaction_base_type'] = main_df['transaction_base_type'].replace(
                to_replace=["debit", "credit"], value=[1, 0])
        except (TypeError, OSError, ValueError) as e:
            print(f"Problem with mem_id/tr_b_type conversion: {e}")

        date_features = ['optimized_transaction_date']
        try:
            for feature in date_features:
                if main_df[feature].isnull().sum() == 0:
                    main_df[feature] = main_df[feature].apply(lambda x: dt.timestamp(x))
                else:
                    main_df = main_df.drop(columns=feature, axis=1)
                    print(f"Column {feature} dropped")
        except (TypeError, OSError, ValueError) as e:
            print(f"Problem with date conversion: {e}")

        encoding_features = ['description', 'primary_merchant_name',
                             'transaction_category_name', 'state',
                             'city', 'transaction_origin']

        UNKNOWN_TOKEN = '<unknown>'
        embedding_maps = {}
        for feature in encoding_features:
            unique_list = main_df[feature].unique().astype('str').tolist()
            unique_list.append(UNKNOWN_TOKEN)
            le = LabelEncoder()
            le.fit_transform(unique_list)
            embedding_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

            # APPLICATION TO DATASET
            main_df[feature] = main_df[feature].apply(lambda x: x if x in embedding_maps[feature] else UNKNOWN_TOKEN)
            main_df[feature] = main_df[feature].map(lambda x: le.transform([x])[0] if type(x) == str else x)

        main_df = main_df.dropna()

        X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, \
            X_test_minmax, y_train, y_test = split_data_feat(df=main_df,
                                                             features=encoding_features,
                                                             test_size=0.30,
                                                             label='primary_merchant_name')

        # convert train data to ndarray to avoid feature_names mismatch error
        X_array = X_train.values
        y_array = y_train.values
        Xt_array = X_test.values
        yt_array = y_test.values

        clf_object = pipeline_xgb(x=X_array,
                                      y=y_array,
                                      test_features=Xt_array,
                                      test_target=yt_array)


        y_pred = clf_object.predict(Xt_array)

        # inverse transformation to merchant strings
        #decoded_merchants = dict(zip(le.classes_, le.inverse_transform(y_pred)))
        #gen_merch = (i for i in decoded_merchants.items())

        # store trained model as pickle
        store_pickle(model=clf_object,
                     file_name=f"trained_model_{user[0]}")


        try:
            connection = create_connection(db_name=db_name,
                                            db_user=db_user,
                                            db_password=db_user,
                                            db_host=db_host,
                                            db_port=db_port)
            print("-------------")
            cursor = connection.cursor()
            sql_insert_query = """
            INSERT INTO test (test_col_2)
            VALUES (%s);
            """
            merch_list = ['Tatte Bakery', 'Star Market', 'Stop n Shop', 'Auto Parts Shop',
                          'Trader Joes', 'Insomnia Cookies']
            # tuple or list works
            for i in merch_list:
            # executemany() to insert multiple rows rows
                cursor.execute(sql_insert_query, (i, ))

            connection.commit()
            print(len(merch_tuple), "record(s) inserted successfully.")

        except (Exception, psycopg2.Error) as error:
            print("Failed inserting record {}".format(error))

        finally:
            # closing database connection.
            if (connection):
                cursor.close()
                connection.close()
                print("Operation accomplished.\nPostgreSQL connection is closed.")
        print("---------------")
    return'done'
#%%

docker_test()

