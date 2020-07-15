from xgboost import XGBClassifier

def pipeline_xgb():

    xgbclf = XGBClassifier()
    # Add silent=True to avoid printing out updates with each cycle
    xgbclf.fit(X_train, y_train, silent=True)

    # make predictions
    y_pred = xgbclf.predict(X_test)
    print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))

    return xgbclf