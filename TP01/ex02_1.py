import pylab as pl
from sklearn.datasets import load_iris
from sklearn import neighbors

irisData=load_iris()
X=irisData.data
Y=irisData.target

nb_voisins = 15

clf = neighbors.KNeighborsClassifier(nb_voisins)
clf.fit(X, Y)

print(clf.score(X,Y))
