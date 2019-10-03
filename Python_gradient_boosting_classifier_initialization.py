#Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingRegressor
def gradient_boosting_cl(alpha, learning_rate, n_estimators, max_depth, random_state):
	#alpha: regularization parameter; the higher the stricter the parameters are forced toward zero
	GBR = GradientBoostingRegressor(alpha = 0.01,learning_rate = 0.01, n_estimators = 150,max_depth = 5 ,random_state = 0)
	GBR.fit(X_train, y_train)

