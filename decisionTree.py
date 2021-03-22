from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics


def decisionTreeAccuracy(X_train, X_test, y_train, y_test, criterion="entropy"):
    # Treinamento da Árvore de Decisão
    model = tree.DecisionTreeClassifier(criterion=criterion)
    model = model.fit(X_train, y_train)

    # Predição e Resultados
    predict = model.predict(X_test)
    acc = metrics.accuracy_score(predict, y_test)
    result = round(acc * 100)

    return result
