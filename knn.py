from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def knnAccuracy(X_train, X_test, y_train, y_test, K, metric="euclidean"):
    model = KNeighborsClassifier(
        n_neighbors=K, metric=metric, algorithm="brute")
    model = model.fit(X_train, y_train)

    predict = model.predict(X_test)
    acc = metrics.accuracy_score(predict, y_test)
    result = round(acc * 100)

    return result
