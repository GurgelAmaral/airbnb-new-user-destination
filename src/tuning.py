from sklearn.model_selection import GridSearchCV

params = {
    'model_pipeline__n_estimators':[100, 200, 300, 400],
    'model_pipeline__max_depth':[None, 10, 20, 30]
}

def get_best_estimators(model, x_train, y_train):
    grid = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=10)
    grid.fit(x_train, y_train)

    return grid.best_estimator_