from sklearn import metrics
import math
from collections import Counter
import numpy


def knnImproveAccuracy(X_train, X_test, y_train, y_test, K):
    # Tranforma os dados em listas
    train_x = X_train.values.tolist()
    train_y = y_train.values.tolist()

    test_x = X_test.values.tolist()
    test_y = y_test.values.tolist()

    resultKNN_improve = []

    raios = calcular_raios(train_x, train_y)

    for i in range(len(test_x)):
        classe = knn_improve(train_x, train_y, test_x[i], K, raios)
        resultKNN_improve.append(classe)

    acc = metrics.accuracy_score(resultKNN_improve, test_y)
    result = round(acc * 100)

    return result


def calcular_raios(train_x, train_y):
    e = 1e-20
    raios = []

    for i in range(len(train_x)):
        newData = train_x.copy()
        newData.pop(i)
        newData_y = train_y.copy()
        newData_y.pop(i)

        results = []

        for j in range(len(newData)):
            r = 0

            for k in range(len(train_x[i])):
                # Distância Euclidiana
                r += (train_x[i][k] - newData[j][k]) ** 2

            results.append(math.sqrt(r))

        indexes = numpy.argsort(results)  # retorna os índices ordenados

        aux = 0
        while train_y[i] == newData_y[indexes[aux]]:
            aux += 1

        raios.append(results[indexes[aux]] - e)

    return raios


def knn_improve(train_x, train_y, test, k, raios):
    results = []

    for i in range(len(train_x)):
        r = 0

        for j in range(len(test)):
            r += (test[j] - train_x[i][j]) ** 2  # Distância Euclidiana

        results.append(math.sqrt(r)/raios[i])  # Distância Euclidiana / Raio

    indexes = numpy.argsort(results)  # retorna os índices ordenados

    indexes = indexes[0:k]  # Pega os k índices mais próximos

    # Retorna a classe de cada um dos vizinhos
    res = [train_y[i] for i in indexes]

    final = Counter(res)

    return final.most_common(1)[0][0]  # retorna a classe com maior frequência
