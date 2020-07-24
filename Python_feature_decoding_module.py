# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:14:32 2020

@author: bill-
"""

"""
Module to apply decoded prediction to the dataframe and fill the missing values
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Python_df_encoder import df_encoder
from Python_split_data_w_features import split_data_feat
from xgbc_class import pipeline_xgb
from Python_store_pickle import store_pickle
from Python_open_pickle import open_pickle
from Python_SQL_connection import insert_val, insert_val_alt, create_connection, execute_read_query
import PostgreSQL_credentials as acc


connection = create_connection(db_name=acc.YDB_name,
                                db_user=acc.YDB_user,
                                db_password=acc.YDB_password,
                                db_host=acc.YDB_host,
                                db_port=acc.YDB_port)

fields = ['unique_mem_id', 'description', 'primary_merchant_name', 'transaction_category_name', 'amount', 'state',
              'city', 'transaction_base_type', 'transaction_origin', 'optimized_transaction_date']
# use these columns as features
# dropped amount_mean_lag7 to avoid errors
#feat_merch = ['unique_mem_id', 'description', 'transaction_category_name', 'amount', 'state',
#              'city', 'transaction_base_type', 'transaction_origin']
# pull data and encode
# section from 1 - 10
section = 8
try:
    filter_query = f"(SELECT {', '.join(field for field in fields)} FROM card_record \
                        WHERE unique_mem_id IN \
                        (SELECT unique_mem_id FROM user_demographic \
                         ORDER BY unique_mem_id ASC limit 10000 offset {10000*(section-1)})) \
                        UNION ALL (SELECT {', '.join(field for field in fields)} \
                       FROM bank_record WHERE unique_mem_id IN \
                       (SELECT unique_mem_id FROM user_demographic \
                        ORDER BY unique_mem_id ASC limit 10000 offset {10000*(section-1)}))"
    transaction_query = execute_read_query(connection, filter_query)
    main_df = pd.DataFrame(transaction_query, columns=fields)
    print(f"{len(transaction_query)} transactions.")
except OperationalError as e:
    print(f"The error '{e}' occurred")
    connection.rollback

for num, user in enumerate(main_df.groupby('unique_mem_id')):
    print(f"User: {user[0]}, {num+1}/10000 users, {round(((num+1)/10000)*100, 2)}%.")

    # create the dict for encoded feature for each user (is overwritten each time)
    encoding_features = ['primary_merchant_name']
    UNKNOWN_TOKEN = '<unknown>'
    embedding_maps = {}
    for feature in encoding_features:
        unique_list = main_df[feature].unique().astype('str').tolist()
        unique_list.append(UNKNOWN_TOKEN)
        le = LabelEncoder()
        le.fit_transform(unique_list)
        embedding_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

    df = df_encoder(df=main_df,
                    spending_report=False,
                    plots=False,
                    include_lag_features=False)

    X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, \
        X_test_minmax, y_train, y_test = split_data_feat(df=main_df,
                                                         features=fields,
                                                         test_size=0.2,
                                                         label='primary_merchant_name')

    # convert train data to ndarray to avoid feature_names mismatch error
    X_array = X_train.values
    y_array = y_train.values
    Xt_array = X_test.values
    yt_array = y_test.values
    # X_train and y_train used to train pipeline
    xgb_clf_object = pipeline_xgb(x=X_array,
                                  y=y_array,
                                  test_features=Xt_array,
                                  test_target=yt_array,
                                  verb=True)

    # array object
    y_pred = xgb_clf_object.predict(Xt_array)
    # inverse transformation to merchant strings
    decoded_merchants = dict(zip(le.classes_, le.inverse_transform(y_pred)))

    # insert query into dataframe (PROBLEM FOR-LOOP in SQL)
    my_sql_string = """test
                    """


    # insert values into Yodlee DB
    # version 1
    #insert_val(query_string) = my_sql_string

    # version 2
    #insert_val_alt(insertion_val = ,
    #               columns = )

    # store trained model as pickle
    store_pickle(model=xgb_clf_object)

    # open the model; located in the current folder
    #trained_model = open_pickle(model_file="gridsearch_model.sav")
