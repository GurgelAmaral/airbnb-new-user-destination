from sklearn.model_selection import GridSearchCV
import time

params = {
    'classifier__n_estimators':[400, 410],
    'classifier__max_depth':[None],
    'classifier__n_jobs':[-1]
}

def get_best_estimators(model, x_train, y_train, cv=5):
    grid = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f'Preparação de busca por hiperparâmetros [CHECK]')
    print('-'*128)

    start = time.time()
    print('Buscando e treinando melhor modelo . . .')
    grid.fit(x_train, y_train.squeeze())
    end = time.time()

    print('Modelo treinado [CHECK]')
    print(f'Tempo decorrido de treino e busca: {end - start} s')
    print('-'*128)
    return grid.best_estimator_