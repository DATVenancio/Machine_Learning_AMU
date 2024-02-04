import random
from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.model_selection import KFold
irisData=load_iris()
X=irisData.data
Y=irisData.target

#kf=KFold(len(X),n_folds=10,shuffle=True)
kf=KFold(n_splits=3,shuffle=False)
scores=[]
for k in range(1,30):
    score=0
    clf = neighbors.KNeighborsClassifier(k)
    #for learn,test in kf:
    for learn,test in kf.split(X):
        X_train=X[learn]
        Y_train=Y[learn]
        clf.fit(X_train, Y_train)
        X_test=X[test]
        Y_test=Y[test]
        score = score + clf.score(X_test,Y_test)
    scores.append(score)
print(["{:4.2f}".format(s) for s in scores])
print("meilleure valeur pour k : ",scores.index(max(scores))+1)

#it goes wrong, the alg to find the best k doesnt work really well