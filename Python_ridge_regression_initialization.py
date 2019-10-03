#Ridge Regression

from sklearn.metrics import score
from sklearn.linear_model import Ridge
#setting values
alpha = 
random_state = 


def ridge_regression:

	Ridge = Ridge(alpha = alpha, random_state = random_state)
	Ridge.fit(X_train, y_train)
	y_test = Ridge.predict(X_test)
	f"Training set accuracy: {Ridge.score(X_train, y_train)}; Test set accuracy: {Ridge.score(X_test, y_test)}"
	return

