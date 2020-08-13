# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:14:32 2020

@author: bill-
"""

"""
Preparations script to apply prediction to the dataframe row by row
Each prediction is decoded from number to string and inserted into the
PostgreSQL database
"""


from yodlee_encoder import df_encoder
from split_data import split_data
from guides.Python_classifier_testing import pipeline_xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from pickle_io import store_pickle, open_pickle
from SQL_connection import insert_val


# use these columns as features
# dropped amount_mean_lag7 to avoid errors
feat_merch = ['description', 'transaction_category_name', 'amount', 'state',
              'city', 'transaction_base_type', 'transaction_origin']

df = df_encoder(rng=14,
                spending_report=False,
                plots=False,
                include_lag_features=False)

X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, \
    X_test_minmax, y_train, y_test = split_data(df= df,
                                                features = feat_merch,
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
                              verb=False)

#df[trained_model.get_booster().feature_names]

# iterate through rows; apply the xgb model to each set of feat
# df needs to boast the same columns as the training data
for index, row in df[feat_merch].iterrows():
    # prediction with model object per row
    y_pred = xgb_clf_object.predict(row)

    # y_pred = trained_model.predict(row)

    # insert query into dataframe (PROBLEM FOR-LOOP in SQL)
    print(y_pred)
#%%
'''
Catching the predictions and converting them back to merchants
Should the prediction turn out to be wrong ask for input by the user
Label needs to be primary_merchant_name
'''
def merch_pred():
    merch_list = []
    for merchant, value in embedding_map_merchants.items():
        for prediction in grid_search.predict(X_test):
            if prediction == value:
                #print(f"Transaction at {merchant}")
                merch_list.append(merchant)
            # else:
            #     print("This merchant could not be recognized by us.\nCan you tell us where you are shopping right now? :)")
            #     merch_list.append("Wrong prediction")
    return merch_list
#%%
# store trained model as pickle
store_pickle(model=xgb_clf_object)

# open the model; located in the current folder
trained_model = open_pickle(model_file="gridsearch_model.sav")
=======
# X_train and y_train used to train pipeline
xgb_clf_object = pipeline_xgb(x=X_train,
                              y=y_train,
                              test_features=X_test,
                              test_target=y_test,
                              verb=False)

# store trained model as pickle
store_pickle(model=xgb_clf_object)
trained_model = open_pickle(model_file="gridsearch_model.sav")


df[trained_model.get_booster().feature_names]
# iterate through rows; apply the xgb model to each set of feat
for index, row in df.iterrows():
    #print(row[feat_merch])
    # prediction with model object per row
    y_pred = xgb_clf_object.predict(row)
    # insert query into dataframe (PROBLEM FOR-LOOP in SQL)
    print(y_pred)
>>>>>>> 2aeb4553f056a1b11bf9809b103ee1825b9d2596
