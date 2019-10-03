#Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingRegressor
def gradient_boosting_cl(alpha, learning_rate, n_estimators, max_depth, random_state):
	#alpha: regularization parameter; the higher the stricter the parameters are forced toward zero
	GBR = GradientBoostingRegressor(alpha = alpha,learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth ,random_state = random_state)
	GBR.fit(X_train, y_train)
	GBR.predict(X_test)
	f"Training set accuracy: {GBR.score(X_train, y_train)}; Test set accuracy: {GBR.score(X_test, y_test)}"
	return

