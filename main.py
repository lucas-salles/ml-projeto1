import pandas as pd
from dataSplit import dataSplit
from decisionTree import decisionTreeAccuracy
from knnImprove import knnImproveAccuracy
from knn import knnAccuracy

wineDataset = pd.read_csv("datasets/wine.data", header=None)
cancerDataset = pd.read_csv("datasets/cancer.csv", header=None)

# X_train, X_test, y_train, y_test de wine dataset
wineX_train, wineX_test, wineY_train, wineY_test = dataSplit(wineDataset)

# X_train, X_test, y_train, y_test de cancer dataset
cancerX_train, cancerX_test, cancerY_train, cancerY_test = dataSplit(
    cancerDataset)


###############
#
# Árvores de decisão
#
###############

# Árvores de decisão para wine dataset com o criterion entropy e gini
wineEntropyAccuracy = decisionTreeAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, "entropy")
wineGiniAccuracy = decisionTreeAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, "gini")

# Árvores de decisão para cancer dataset com o criterion entropy e gini
cancerEntropyAccuracy = decisionTreeAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, "entropy")
cancerGiniAccuracy = decisionTreeAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, "gini")

print("=-=-=-=-=-=-=-= Árvores de Decisão =-=-=-=-=-=-=-=")

print("Wine decision tree with entropy: {}%".format(wineEntropyAccuracy))
print("Wine decision tree with gini: {}%".format(wineGiniAccuracy))

print("Cancer decision tree with entropy: {}%".format(cancerEntropyAccuracy))
print("Cancer decision tree with gini: {}%".format(cancerGiniAccuracy))

print()


###############
#
# KNN
#
###############

# KNN com métricas euclidean e manhattan e k = 3 para wine dataset
wineKnnEuclideanK3Accuracy = knnAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, 3, "euclidean")
wineKnnManhattanK3Accuracy = knnAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, 3, "manhattan")

# KNN com métricas euclidean e manhattan e k = 7 para wine dataset
wineKnnEuclideanK7Accuracy = knnAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, 7, "euclidean")
wineKnnManhattanK7Accuracy = knnAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, 7, "manhattan")

# KNN com métricas euclidean e manhattan e k = 10 para wine dataset
wineKnnEuclideanK10Accuracy = knnAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, 10, "euclidean")
wineKnnManhattanK10Accuracy = knnAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, 10, "manhattan")

# KNN com métricas euclidean e manhattan e k = 3 para cancer dataset
cancerKnnEuclideanK3Accuracy = knnAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, 3, "euclidean")
cancerKnnManhattanK3Accuracy = knnAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, 3, "manhattan")

# KNN com métricas euclidean e manhattan e k = 7 para cancer dataset
cancerKnnEuclideanK7Accuracy = knnAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, 7, "euclidean")
cancerKnnManhattanK7Accuracy = knnAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, 7, "manhattan")

# KNN com métricas euclidean e manhattan e k = 10 para cancer dataset
cancerKnnEuclideanK10Accuracy = knnAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, 10, "euclidean")
cancerKnnManhattanK10Accuracy = knnAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, 10, "manhattan")

print("=-=-=-=-=-=-=-= KNN =-=-=-=-=-=-=-=")

print("Wine KNN with euclidean metric and K = 3: {}%".format(
    wineKnnEuclideanK3Accuracy))
print("Wine KNN with manhattan metric and K = 3: {}%".format(
    wineKnnManhattanK3Accuracy))
print("Wine KNN with euclidean metric and K = 7: {}%".format(
    wineKnnEuclideanK7Accuracy))
print("Wine KNN with manhattan metric and K = 7: {}%".format(
    wineKnnManhattanK7Accuracy))
print("Wine KNN with euclidean metric and K = 10: {}%".format(
    wineKnnEuclideanK10Accuracy))
print("Wine KNN with manhattan metric and K = 10: {}%".format(
    wineKnnManhattanK10Accuracy))

print("Cancer KNN with euclidean metric and K = 3: {}%".format(
    cancerKnnEuclideanK3Accuracy))
print("Cancer KNN with manhattan metric and K = 3: {}%".format(
    cancerKnnManhattanK3Accuracy))
print("Cancer KNN with euclidean metric and K = 7: {}%".format(
    cancerKnnEuclideanK7Accuracy))
print("Cancer KNN with manhattan metric and K = 7: {}%".format(
    cancerKnnManhattanK7Accuracy))
print("Cancer KNN with euclidean metric and K = 10: {}%".format(
    cancerKnnEuclideanK10Accuracy))
print("Cancer KNN with manhattan metric and K = 10: {}%".format(
    cancerKnnManhattanK10Accuracy))

print()


###############
#
# KNN Improve
#
###############

# KNN Improve com k = 3 para wine dataset
wineKnnImproveK3Accuracy = knnImproveAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, 3)

# KNN Improve com k = 7 para wine dataset
wineKnnImproveK7Accuracy = knnImproveAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, 7)

# KNN Improve com k = 10 para wine dataset
wineKnnImproveK10Accuracy = knnImproveAccuracy(
    wineX_train, wineX_test, wineY_train, wineY_test, 10)

# KNN Improve com k = 3 para cancer dataset
cancerKnnImproveK3Accuracy = knnImproveAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, 3)

# KNN Improve com k = 7 para cancer dataset
cancerKnnImproveK7Accuracy = knnImproveAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, 7)

# KNN Improve com k = 10 para cancer dataset
cancerKnnImproveK10Accuracy = knnImproveAccuracy(
    cancerX_train, cancerX_test, cancerY_train, cancerY_test, 10)

print("=-=-=-=-=-=-=-= KNN Improve =-=-=-=-=-=-=-=")

print("Wine KNN Improve with K = 3: {}%".format(wineKnnImproveK3Accuracy))
print("Wine KNN Improve with K = 7: {}%".format(wineKnnImproveK7Accuracy))
print("Wine KNN Improve with K = 10: {}%".format(wineKnnImproveK10Accuracy))

print("Cancer KNN Improve with K = 3: {}%".format(
    cancerKnnImproveK3Accuracy))
print("Cancer KNN Improve with K = 7: {}%".format(
    cancerKnnImproveK7Accuracy))
print("Cancer KNN Improve with K = 10: {}%".format(
    cancerKnnImproveK10Accuracy))
