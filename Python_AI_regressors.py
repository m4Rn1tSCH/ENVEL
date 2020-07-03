# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:11:55 2020

@author: bill-
"""

# load packages
from sklearn.feature_selection import SelectKBest , chi2, f_classif, RFE, RFECV
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LogisticRegression,SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
#####################################################

# @app.route('/pipeline_sgd_reg')
def pipeline_sgd_reg():

    '''
    SelectKBest and SGDRegressor -needs non-negative values

    '''
    #Create pipeline with feature selector and regressor
    #replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('reg', SGDRegressor(loss='squared_loss', penalty='l1'))
        ])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERS NEED TO HAVE THE SAME LENGTH
    params = {
        'feature_selection__k':[5, 6, 7],
        'reg__alpha':[0.01, 0.001, 0.0001],
        'reg__max_iter':[800, 1000, 1500]
        }

    #Initialize the grid search object
    grid_search_sgd = GridSearchCV(pipe, param_grid = params)

    #Fit it to the data and print the best value combination
    print(f"Pipeline sgd-reg; {dt.today()}")
    print(grid_search_sgd.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_sgd.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_sgd.best_score_}")

    return grid_search_sgd
#%%
# @app.route('/pipeline_rfr')
def pipeline_rfr():

    '''
    SelectKBest and Random Forest Regressor
    '''
    #Create pipeline with feature selector and classifier
    #replace with gradient boosted at this point or regessor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('reg', RandomForestRegressor(n_estimators = 75,
                                      max_depth = len(bank_df.columns)/2,
                                      min_samples_split = 4))
        ])

    #Create a parameter grid
    #parameter grids provide the values for the models to try
    #PARAMETERS NEED TO HAVE THE SAME LENGTH
    params = {
        'feature_selection__k':[5, 6, 7, 8, 9],
        'reg__n_estimators':[75, 100, 150, 200],
        'reg__min_samples_split':[4, 8, 10, 15],
        }

    # Initialize the grid search object
    grid_search_rfr = GridSearchCV(pipe, param_grid = params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline rfr; {dt.today()}")
    print(grid_search_rfr.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_rfr.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_rfr.best_score_}")

    return grid_search_rfr
#%%
# @app.route('/pipeline_svr')
def pipeline_svr():

    '''
    Logistic Regression and Support Vector Kernel -needs non-negative values
    '''
    #Create pipeline with feature selector and regressor
    #replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('reg', SVR(kernel = 'linear'))
        ])

    # Create a parameter grid

    # C regularization parameter that is applied to all terms
    # to push down their individual impact and reduce overfitting
    # Epsilon tube around actual values; threshold beyond which regularization is applied
    # the more features picked the more prone the model is to overfitting
    # stricter C and e to counteract
    params = {
        'feature_selection__k':[4, 6, 7, 8, 9],
        'reg__C':[1.0, 0.1, 0.01, 0.001],
        'reg__epsilon':[0.30, 0.25, 0.15, 0.10],
        }

    # Initialize the grid search object
    grid_search_svr = GridSearchCV(pipe, param_grid = params)

    # best combination of feature selector and the regressor
    # grid_search.best_params_
    # best score
    # grid_search.best_score_
    # Fit it to the data and print the best value combination
    print(f"Pipeline svr; {dt.today()}")
    print(grid_search_svr.fit(X_train_minmax, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_svr.score(X_test_minmax, y_test)))
    print(f"Best accuracy with parameters: {grid_search_svr.best_score_}")

    return grid_search_svr