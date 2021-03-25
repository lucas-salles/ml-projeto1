import pandas as pd
from dataSplit import dataSplit
from decisionTree import decisionTreeAccuracy
from knnImprove import knnImproveAccuracy
from knn import knnAccuracy

wineDataset = pd.read_csv("datasets/wine.data", header=None)
glassDataset = pd.read_csv("datasets/glass.data", header=None)

# X_train, X_test, y_train, y_test de wine dataset
wineX_train, wineX_test, wineY_train, wineY_test = dataSplit(wineDataset)

# X_train, X_test, y_train, y_test de glass dataset
glassX_train, glassX_test, glassY_train, glassY_test = dataSplit(
    glassDataset)


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

# Árvores de decisão para glass dataset com o criterion entropy e gini
glassEntropyAccuracy = decisionTreeAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, "entropy")
glassGiniAccuracy = decisionTreeAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, "gini")

print("=-=-=-=-=-=-=-= Árvores de Decisão =-=-=-=-=-=-=-=")

print("Wine decision tree with entropy: {}%".format(wineEntropyAccuracy))
print("Wine decision tree with gini: {}%".format(wineGiniAccuracy))

print("Glass decision tree with entropy: {}%".format(glassEntropyAccuracy))
print("Glass decision tree with gini: {}%".format(glassGiniAccuracy))

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

# KNN com métricas euclidean e manhattan e k = 3 para glass dataset
glassKnnEuclideanK3Accuracy = knnAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, 3, "euclidean")
glassKnnManhattanK3Accuracy = knnAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, 3, "manhattan")

# KNN com métricas euclidean e manhattan e k = 7 para glass dataset
glassKnnEuclideanK7Accuracy = knnAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, 7, "euclidean")
glassKnnManhattanK7Accuracy = knnAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, 7, "manhattan")

# KNN com métricas euclidean e manhattan e k = 10 para glass dataset
glassKnnEuclideanK10Accuracy = knnAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, 10, "euclidean")
glassKnnManhattanK10Accuracy = knnAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, 10, "manhattan")

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

print("Glass KNN with euclidean metric and K = 3: {}%".format(
    glassKnnEuclideanK3Accuracy))
print("Glass KNN with manhattan metric and K = 3: {}%".format(
    glassKnnManhattanK3Accuracy))
print("Glass KNN with euclidean metric and K = 7: {}%".format(
    glassKnnEuclideanK7Accuracy))
print("Glass KNN with manhattan metric and K = 7: {}%".format(
    glassKnnManhattanK7Accuracy))
print("Glass KNN with euclidean metric and K = 10: {}%".format(
    glassKnnEuclideanK10Accuracy))
print("Glass KNN with manhattan metric and K = 10: {}%".format(
    glassKnnManhattanK10Accuracy))

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

# KNN Improve com k = 3 para glass dataset
glassKnnImproveK3Accuracy = knnImproveAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, 3)

# KNN Improve com k = 7 para glass dataset
glassKnnImproveK7Accuracy = knnImproveAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, 7)

# KNN Improve com k = 10 para glass dataset
glassKnnImproveK10Accuracy = knnImproveAccuracy(
    glassX_train, glassX_test, glassY_train, glassY_test, 10)

print("=-=-=-=-=-=-=-= KNN Improve =-=-=-=-=-=-=-=")

print("Wine KNN Improve with K = 3: {}%".format(wineKnnImproveK3Accuracy))
print("Wine KNN Improve with K = 7: {}%".format(wineKnnImproveK7Accuracy))
print("Wine KNN Improve with K = 10: {}%".format(wineKnnImproveK10Accuracy))

print("Glass KNN Improve with K = 3: {}%".format(
    glassKnnImproveK3Accuracy))
print("Glass KNN Improve with K = 7: {}%".format(
    glassKnnImproveK7Accuracy))
print("Glass KNN Improve with K = 10: {}%".format(
    glassKnnImproveK10Accuracy))
