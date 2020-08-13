from sklearn.preprocessing import LabelEncoder
from psycopg2 import OperationalError
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import pandas as pd
from datetime import datetime as dt

# FILE IMPORTS FOR NOTEBOOKS
from ml_code.model_data.SQL_connection import execute_read_query, create_connection
import ml_code.model_data.PostgreSQL_credentials as acc
from ml_code.classification_models.svc_class import svc_class

# extensions to convert numpy objects to recognizable SQL objects
def adapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

def adapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

def adapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

def adapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))

register_adapter(np.float64, adapt_numpy_float64)
register_adapter(np.int64, adapt_numpy_int64)
register_adapter(np.float32, adapt_numpy_float32)
register_adapter(np.int32, adapt_numpy_int32)
register_adapter(np.ndarray, adapt_numpy_array)


# query user
# encode fields keep embedding_map for later
# split into df and null_df

def df_encoder(state='NV', sel_feature='state'):
    '''
    Returns
    -------
    [df, df_null, embedding_maps[sel_feature]]
    '''

    connection = create_connection(db_name=acc.YDB_name,
                                   db_user=acc.YDB_user,
                                   db_password=acc.YDB_password,
                                   db_host=acc.YDB_host,
                                   db_port=acc.YDB_port)

    states = ['NJ', 'AL', 'WA', 'NM', 'TX', 'AR', 'NY', 'CT', 'MI', 'CO', 'CA',
              'PA', 'IN', 'OK', 'MD', 'AK', 'VA', 'GA', 'NC', 'TN', 'OH', 'IL',
              'FL', 'AZ', 'DC', 'LA', 'KY', 'KS', 'IA', 'SC', 'WI', 'DE', 'HI',
              'MT', 'MO', 'NV', 'ID', 'MN', 'MS', 'OR', 'UT', 'NH', 'MA', 'WV',
              'NE', 'ND', 'RI', 'VT', 'WY', 'ME', 'SD', 'PR', 'GU']

    fields = ['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id', 
              'amount', 'currency', 'description', 'transaction_date', 
             'transaction_base_type', 'transaction_category_name', 'primary_merchant_name', 
              'city', 'state', 'transaction_origin', 'optimized_transaction_date', 
              'account_type', 'account_source_type', 'account_score', 'user_score', 
              'panel_file_created_date']

    try:
        if state in states:
            filter_query = f"SELECT {', '.join(field for field in fields)} FROM bank_record WHERE unique_mem_id in ( SELECT unique_mem_id FROM user_demographic WHERE state = '{state}' limit 1)"
            transaction_query = execute_read_query(connection, filter_query)
            df = pd.DataFrame(transaction_query,
                                   columns=fields)
            print(f"{len(df)} transactions for random user in state: {state}.")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        connection.rollback

    '''
    Generate a spending report of the unaltered dataframe
    Use the datetime columns just defined
    This report measures either the sum or mean of transactions happening
    on various days of the week/or wihtin a week or a month  over the course of the year
    '''
    # convert all date col from date to datetime objects
    # date objects will block Select K Best if not converted
    # first conversion from date to datetime objects; then conversion to unix
    df['panel_file_created_date'] = pd.to_datetime(
        df['panel_file_created_date'])
    df['optimized_transaction_date'] = pd.to_datetime(
        df['optimized_transaction_date'])
    df['transaction_date'] = pd.to_datetime(
        df['transaction_date'])

    # set optimized transaction_date as index for later
    df.set_index('optimized_transaction_date', drop=False, inplace=True)

    '''
    After successfully loading the data, columns that are of no importance have been removed and missing values replaced
    Then the dataframe is ready to be encoded to get rid of all non-numerical data
    '''
    try:
        # Include below if need unique ID's later:
        df['unique_mem_id'] = df['unique_mem_id'].astype(
            'str', errors='ignore')
        df['unique_bank_account_id'] = df['unique_bank_account_id'].astype(
            'str', errors='ignore')
        df['unique_bank_transaction_id'] = df['unique_bank_transaction_id'].astype(
            'str', errors='ignore')
        df['amount'] = df['amount'].astype('float64')
        df['transaction_base_type'] = df['transaction_base_type'].replace(
            to_replace=["debit", "credit"], value=[1, 0])
    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    # attempt to convert date objects to unix timestamps as numeric value (fl64) if they have no missing values; otherwise they are being dropped
    date_features = ['optimized_transaction_date', 'panel_file_created_date', 'transaction_date']
    try:
        for feature in date_features:
            if df[feature].isnull().sum() == 0:
                df[feature] = df[feature].apply(lambda x: dt.timestamp(x))
            else:
                df = df.drop(columns=feature, axis=1)
                print(f"Column {feature} dropped")

    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    '''
    The columns PRIMARY_MERCHANT_NAME; CITY, STATE, DESCRIPTION, TRANSACTION_CATEGORY_NAME, CURRENCY
    are encoded manually and cleared of empty values
    '''
    encoding_features = ['primary_merchant_name', 'city', 'state', 'description', 
                        'transaction_category_name', 'transaction_origin', 'currency']

    # dropping currency if there is only one
    if len(df['currency'].value_counts()) == 1:
        df = df.drop(columns=['currency'], axis=1)
        encoding_features.remove('currency')

    UNKNOWN_TOKEN = '<unknown>'
    embedding_maps = {}
    for feature in encoding_features:
        unique_list = df[feature].unique().astype('str').tolist()
        unique_list.append(UNKNOWN_TOKEN)
        le = LabelEncoder()
        le.fit_transform(unique_list)
        embedding_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

        # APPLICATION TO OUR DATASET
        df[feature] = df[feature].apply(lambda x: x if x in embedding_maps[feature] else UNKNOWN_TOKEN)
        df[feature] = df[feature].map(lambda x: le.transform([x])[0] if type(x) == str else x)

    #drop all features left with empty (NaN) values
    df = df.dropna()

    # extract rows with empty value of selected feature into own db
    df_null = df[df[sel_feature] == 0]
    df = df[df[sel_feature] != 0]

    return [df, df_null, embedding_maps[sel_feature]]


from sklearn.model_selection import train_test_split

# df without any missing values
# df_null with missing values in feat or label column



def split_data(full_df, null_df, label='state'):
    model_features = full_df.drop(labels=label, axis=1)
    model_label = full_df[label]

    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        random_state=7,
                                                        shuffle=True,
                                                        test_size=0.2)

    X_features = null_df.drop(labels=label, axis=1)

    return [X_features, X_test, X_train, y_test, y_train]



def add_pred(grid_search, X_features, label='state'):
    predictions = grid_search.predict(X_features)

    pred_df = X_features
    pred_df[label] = predictions
    
    return pred_df, predictions

#%%
# running functions in order
df, df_null, embedding_map = df_encoder()
X_features, X_test, X_train, y_test, y_train = split_data(full_df=df,
                                                          null_df=df_null)
grid_search_svc = svc_class(X_train, X_test, y_train, y_test)
prediction_df, predictions = add_pred(grid_search=grid_search_svc,
                         X_features=X_features, label='state')

# pred_df = pd.DataFrame(data=predictions)
# unseen = '<not in training>'
# not_found = [i for i in predictions if i not in embedding_map.values()]
# dec_pred = pred_df.replace(not_found, unseen)
register_adapter(np.int64, adapt_numpy_int64)

merchants = []
# resulting variable is an array!



for val, enc in embedding_map.items():
    if enc in predictions:
        merchants.append(val)
    else:
        merchants.append("unseen in training")



db_name = "postgres"
db_user = "envel"
db_pw = "envel"
db_host = "0.0.0.0"
db_port = "5432"

'''
Always use %s placeholder for queries; psycopg2 will convert most data automatically
'''

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
    # merch_list = np.ndarray(['Tatte Bakery', 'Star Market', 'Stop n Shop', 'Auto Parts Shop',
    #               'Trader Joes', 'Insomnia Cookies'])


    # values = tuple([tuple(row) for row in merchants])
    values = merchants
    for i in values:
    # executemany() to insert multiple rows rows
    # one-element-tuple with (i, )
        cursor.execute(sql_insert_query, (i, ))

    connection.commit()
    print(len(values), "record(s) inserted successfully.")

except (Exception, psycopg2.Error) as error:
    print("Failed inserting record; {}".format(error))

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("Operation completed.\nPostgreSQL connection is closed.")
print("=========================")


