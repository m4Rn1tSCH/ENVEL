from sklearn.ensemble import RandomForestClassifier


def random_forest_cl(max_depth, max_features, n_estimators, random_state, n_jobs):
	RFC = RandomForestClassifier(max_depth = max_depth, max_features = max_features, n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs)
	RFC.fit(X_train, y_train)

	y_test = RFC.predict(X_test)
	print(y_test)
	
	#evaluate accuracy of RandomForestClassifier
	f"Training set accuracy: {RFC.score(X_train, y_train)}; Test set accuracy: {RFC.score(X_test, y_test)}

