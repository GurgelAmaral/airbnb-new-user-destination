from sklearn.metrics import accuracy_score

def evaluate_model(y_test, y_pred):
    print('Calculando acurácia . . .')
    return accuracy_score(y_test, y_pred)