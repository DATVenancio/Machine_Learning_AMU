import pylab as pl # permet de remplacer le nom "pylab" par "pl"
from sklearn.datasets import load_iris

irisData=load_iris()
X=irisData.data
Y=irisData.target
print(irisData.feature_names)
x = 2
y = 3
colors=["red","green","blue"]
for i in range(3):
    pl.scatter(X[Y==i][:, x],X[Y==i][:,y],color=colors[i],label=irisData.target_names[i])
pl.legend()
pl.xlabel(irisData.feature_names[x])
pl.ylabel(irisData.feature_names[y])
pl.title(u"Donn´ees Iris - dimension des s´epales uniquement")
pl.show()

#Result: the best couple of atributs are petal width and petal length