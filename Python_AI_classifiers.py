#  -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:08:14 2020

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
# @app.route('/pipeline_logreg')
def pipeline_logreg():

    '''
                Setting up a pipeline
    Pipeline 1 - SelectKBest and Logistic Regression (non-neg only)

    f_classif for classification tasks
    chi2 for regression tasks

    SelectKBest picks features based on their f-value to find the features that can optimally predict the labels
    F_CLASSIFIER;FOR CLASSIFICATION TASKS determines features based on the f-values between features & labels;
    Chi2: for regression tasks; requires non-neg values
    other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression
    '''

    # Create pipeline with feature selector and regressor
    # replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('reg', LogisticRegression(random_state = 15))])

    # Create a parameter grid
    # parameter grids provide the values for the models to try
    # PARAMETERS NEED TO HAVE THE SAME LENGTH
    params = {
        'feature_selection__k':[5, 6, 7, 8, 9],
        'reg__max_iter':[800, 1000],
        'reg__C':[10, 1, 0.1]
        }

    # Initialize the grid search object
    grid_search_lr = GridSearchCV(pipe, param_grid = params)

    # best combination of feature selector and the regressor
    # grid_search.best_params_
    # best score
    # grid_search.best_score_

    # Fit it to the data and print the best value combination
    print(f"Pipeline logreg; {dt.today()}")
    print(grid_search_lr.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_lr.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_lr.best_score_}")

    return grid_search_lr
# %%
# @app.route('/pipeline_knn')
def pipeline_knn():

    '''
    Pipeline 6 - SelectKBest and K Nearest Neighbor
    '''
    # Create pipeline with feature selector and classifier
    # replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('clf', KNeighborsClassifier())])

    # Create a parameter grid
    # parameter grids provide the values for the models to try
    # PARAMETERS NEED TO HAVE THE SAME LENGTH
    params = {
        'feature_selection__k':[1, 2, 3, 4, 5, 6, 7],
        'clf__n_neighbors':[2, 3, 4, 5, 6, 7, 8]}

    # Initialize the grid search object
    grid_search_knn = GridSearchCV(pipe, param_grid = params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline knn; {dt.today()}")
    print(grid_search_knn.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_knn.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_knn.best_score_}")
    return grid_search_knn
# %%
# @app.route('/pipeline_svc')
def pipeline_svc():

    '''
    SelectKBest and Support Vector Classifier
    '''
    # Create pipeline with feature selector and classifier
    # replace with classifier or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('clf', SVC())])

    # Create a parameter grid
    # parameter grids provide the values for the models to try
    # PARAMETERS NEED TO HAVE THE SAME LENGTH
    # Parameter explanation
    #    C: penalty parameter
    #    gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    #
    params = {
        'feature_selection__k':[4, 5, 6, 7, 8, 9],
        'clf__C':[0.01, 0.1, 1, 10],
        'clf__gamma':[0.1, 0.01, 0.001]}

    # Initialize the grid search object
    grid_search_svc = GridSearchCV(pipe, param_grid = params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline 7; {dt.today()}")
    print(grid_search_svc.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_svc.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_svc.best_score_}")
    return grid_search_svc
# %%
# @app.route('/pipeline_gbc')
def pipeline_gbc():

    '''
    Pipeline 5 - SelectKBest and Gradient Boosting Classifier
    '''
    # Create pipeline with feature selector and classifier
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('clf', GradientBoostingClassifier(random_state = 42))])

    # Create a parameter grid
    params = {
        'feature_selection__k':[1, 2, 3, 4, 5, 6, 7],
        'clf__n_estimators':[15, 25, 50, 75, 120, 200, 350]}

    # Initialize the grid search object
    grid_search = GridSearchCV(pipe, param_grid = params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline 5; {dt.today()}")
    print(grid_search.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search.best_score_}")
# %%
#  @app.route('/pipeline_mlp')
def pipeline_mlp():

    '''
    Pipeline 8 - SelectKBest and Multi-Layer Perceptron
    '''
    # Create pipeline with feature selector and classifier
    # replace with classifier or regressor
    # learning_rate = 'adaptive'; when solver='sgd'
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('clf', MLPClassifier(activation='relu',
                              solver='lbfgs',
                              learning_rate='constant'))])

    # Create a parameter grid
    # Parameter explanation
    #    C: penalty parameter
    #    gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    #
    params = {
        'feature_selection__k':[4, 5, 6, 7],
        'clf__max_iter':[1500, 2000],
        'clf__alpha':[0.0001, 0.001]}

    # Initialize the grid search object
    grid_search_mlp = GridSearchCV(pipe, param_grid = params)

    # Fit it to the data and print the best value combination
    print(f"Pipeline 7; {dt.today()}")
    print(grid_search_mlp.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_mlp.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_mlp.best_score_}")
    return grid_search_mlp