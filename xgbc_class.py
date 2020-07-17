from xgboost import XGBClassifier

def pipeline_xgb(x, y, test_features, test_target, silent=True):

    '''
    x: df/ndarray. Pass training data features.
    y: df/ndarray. Pass training data label that is to be predicted.
    test_features: df/ndarray. Test data features.
    test_target: df/ndarray. Test data label that is to be predicted.
    silent: bool. Print out messages at each stage. Default is True.
    '''

    xgbclf = XGBClassifier()
    # Add silent=True to avoid printing out updates with each cycle
    xgbclf.fit(x, y, silent=True)

    # make predictions
    y_pred = xgbclf.predict(test_features)
    print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, test_target)))

    return xgbclf