from sklearn.preprocessing import LabelEncoder
from psycopg2 import OperationalError
import numpy as np
import pandas as pd
from datetime import datetime as dt

# FILE IMPORTS FOR NOTEBOOKS
from SQL_connection import execute_read_query, create_connection
import PostgreSQL_credentials as acc

def df_encoder(state='NV', sel_feature='state'):
    '''
    Returns
    -------
    [bank_df, bank_df_null, embedding_maps[sel_feature]]
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
            filter_query = f"select {', '.join(field for field in fields)} from bank_record where unique_mem_id in ( select unique_mem_id from user_demographic where state = '{state}' limit 1)"
            transaction_query = execute_read_query(connection, filter_query)
            bank_df = pd.DataFrame(transaction_query,
                                   columns=fields)
            print(f"{len(bank_df)} transactions for random user in state: {state}.")
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
    bank_df['panel_file_created_date'] = pd.to_datetime(
        bank_df['panel_file_created_date'])
    bank_df['optimized_transaction_date'] = pd.to_datetime(
        bank_df['optimized_transaction_date'])
    bank_df['transaction_date'] = pd.to_datetime(
        bank_df['transaction_date'])

    # set optimized transaction_date as index for later
    bank_df.set_index('optimized_transaction_date', drop=False, inplace=True)

    '''
    After successfully loading the data, columns that are of no importance have been removed and missing values replaced
    Then the dataframe is ready to be encoded to get rid of all non-numerical data
    '''
    try:
        # Include below if need unique ID's later:
        bank_df['unique_mem_id'] = bank_df['unique_mem_id'].astype(
            'str', errors='ignore')
        bank_df['unique_bank_account_id'] = bank_df['unique_bank_account_id'].astype(
            'str', errors='ignore')
        bank_df['unique_bank_transaction_id'] = bank_df['unique_bank_transaction_id'].astype(
            'str', errors='ignore')
        bank_df['amount'] = bank_df['amount'].astype('float64')
        bank_df['transaction_base_type'] = bank_df['transaction_base_type'].replace(
            to_replace=["debit", "credit"], value=[1, 0])
    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    # attempt to convert date objects to unix timestamps as numeric value (fl64) if they have no missing values; otherwise they are being dropped
    date_features = ['optimized_transaction_date', 'panel_file_created_date', 'transaction_date']
    try:
        for feature in date_features:
            if bank_df[feature].isnull().sum() == 0:
                bank_df[feature] = bank_df[feature].apply(lambda x: dt.timestamp(x))
            else:
                bank_df = bank_df.drop(columns=feature, axis=1)
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
    if len(bank_df['currency'].value_counts()) == 1:
        bank_df = bank_df.drop(columns=['currency'], axis=1)
        encoding_features.remove('currency')

    UNKNOWN_TOKEN = '<unknown>'
    embedding_maps = {}
    for feature in encoding_features:
        unique_list = bank_df[feature].unique().astype('str').tolist()
        unique_list.append(UNKNOWN_TOKEN)
        le = LabelEncoder()
        le.fit_transform(unique_list)
        embedding_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

        # APPLICATION TO OUR DATASET
        bank_df[feature] = bank_df[feature].apply(lambda x: x if x in embedding_maps[feature] else UNKNOWN_TOKEN)
        bank_df[feature] = bank_df[feature].map(lambda x: le.transform([x])[0] if type(x) == str else x)

    #drop all features left with empty (NaN) values
    bank_df = bank_df.dropna()

    # extract rows with empty value of selected feature into own db
    bank_df_null = bank_df[bank_df[sel_feature] == 0]
    bank_df = bank_df[bank_df[sel_feature] != 0]

    return [bank_df, bank_df_null, embedding_maps[sel_feature]]


from sklearn.model_selection import train_test_split

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


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

def svc_class(X_train, X_test, y_train, y_test):
    '''
    SelectKBest and Support Vector Classifier
    '''
    #Create pipeline with feature selector and classifier
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', SVC())])

    #Create a parameter grid, provide the values for the models to try
    #Parameter explanation:
    #   C: penalty parameter
    #   gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    params = {
        'feature_selection__k': [4, 5, 6, 7, 8, 9],
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__gamma': [0.1, 0.01, 0.001]}

    #Initialize the grid search object
    grid_search_svc = GridSearchCV(pipe, param_grid=params, n_jobs=-1, verbose=2)
    grid_search_svc.fit(X_train, y_train)

    #Print the best value combination
    print(f"Best paramaters: {grid_search_svc.best_params_}")
    print("Overall score: %.4f" % (grid_search_svc.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_svc.best_score_}")

    return grid_search_svc

def add_pred(grid_search, X_features, label='state'):
    predictions = grid_search.predict(X_features)

    pred_df = X_features
    pred_df[label] = predictions
    
    return pred_df


# running functions in order
bank_df, bank_df_null, embedding_map = df_encoder()
X_features, X_test, X_train, y_test, y_train = split_data(full_df=bank_df, null_df=bank_df_null)
grid_search_svc = svc_class(X_train, X_test, y_train, y_test)
add_pred(grid_search=grid_search_svc, X_features=X_features, label='state')
