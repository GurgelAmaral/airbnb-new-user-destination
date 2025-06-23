from sklearn.model_selection import GridSearchCV

params = {
    'classifier__n_estimators':[200, 300, 400],
    'classifier__max_depth':[10, 20, 30]
}

def get_best_estimators(model, x_train, y_train, cv=5):
    grid = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
    grid.fit(x_train, y_train)

    return grid.best_estimator_