# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:14:32 2020

@author: bill-
"""

"""
Preparations script to apply prediction to the dataframe row by row"""


from Python_df_encoder_user import df_encoder
from Python_df_encoder_user import split_data
from Python_classifier_testing import pipeline_xgb
from Python_store_pickle import store_pickle
from Python_open_pickle import open_pickle


# use these columns as features
# dropped amount_mean_lag7 to avoid errors
feat_merch = ['description', 'transaction_category_name', 'amount', 'state',
              'city', 'transaction_base_type', 'transaction_origin']
# pull data and encode
df = df_encoder(rng=14,
                spending_report=False,
                plots=False,
                include_lag_features=False)
# just pull raw data
df_enc = pull_df(rng=14,
                 spending_report=False,
                 plots=False)

# create the dict for encoded feature
encoding_features = ['primary_merchant_name']
UNKNOWN_TOKEN = '<unknown>'
embedding_maps = {}
for feature in encoding_features:
    unique_list = df_enc[feature].unique().astype('str').tolist()
    unique_list.append(UNKNOWN_TOKEN)
    le = LabelEncoder()
    le.fit_transform(unique_list)
    embedding_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

for key, value in embedding_maps.items():
    print(key, value)

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
                              verb=True)

#df[trained_model.get_booster().feature_names]

# generator statement; generated once and exhausted
dec = [embedding_maps[k] for k in sorted(embedding_maps.keys()) if k in y_pred]
merch_val = [v for v in embedding_maps.values()]

test = {key:value for key, value in embedding_maps.items()}
# iterate through rows; apply the xgb model to each set of feat
# df needs to boast the same columns as the training data
for index, row in df[feat_merch].iterrows():
    # prediction with model object per row
    y_pred = xgb_clf_object.predict(row)

    # y_pred = trained_model.predict(row)

    # insert query into dataframe (PROBLEM FOR-LOOP in SQL)
    print("Index: ", index, "Prediction: ", y_pred, "", dec)
#%%
'''
Catching the predictions and converting them back to merchants
Should the prediction turn out to be wrong ask for input by the user
Label needs to be primary_merchant_name
'''
def merch_pred():


    dec = [embedding_maps[k] for k in sorted(embedding_maps.keys())]
    merch_val = [v for v in embedding_maps.values()]

    test = {key:value for key, value in embedding_maps.items()}
merch_list = []
for i in dec:
    if y_pred == value:
        #print(f"Transaction at {merchant}")
            merch_list.append(merchant)

    return merch_list

#%%
# store trained model as pickle
store_pickle(model=xgb_clf_object)

# open the model; located in the current folder
trained_model = open_pickle(model_file="gridsearch_model.sav")
#%%
# BUGGED
# generate dict with merchants
def feature_dict(feature):

    data = df_encoder(rng=14,
                spending_report=False,
                plots=False,
                include_lag_features=False)
    # take feature and convert it to a dictionary
    feature = 'primary_merchant_name'
    unique_list = data[feature].unique().astype('str').tolist()
    UNKNOWN_TOKEN = "<unknown>"
    unique_list.append(UNKNOWN_TOKEN)
    LabelEncoder().fit_transform(unique_list)

    # dict with original unique permutations as keys and transformed as values
    val_dict = {}
    val_dict = dict(zip(LabelEncoder().fit_transform(unique_list), LabelEncoder().inverse_transform())
    yield val_dict

