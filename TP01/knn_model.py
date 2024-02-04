from sklearn.datasets import load_digits
import pylab as pl
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random

dataset=load_digits()
data = dataset.data
label = dataset.target

data_train, data_test, label_train, label_test = train_test_split(data,label,test_size=0.3,random_state=random.seed())


cross_validation_kfold=KFold(n_splits=10,shuffle=True)
scores=[]
for neighbor_count in range(1,30):
    score=0
    neighbors_classifier = neighbors.KNeighborsClassifier(neighbor_count)
    for learn,test in cross_validation_kfold.split(data):
        data_train=data[learn]
        label_train=label[learn]
        data_test=data[test]
        label_test=label[test]

        neighbors_classifier.fit(data_train, label_train)
        score = score + neighbors_classifier.score(data_test,label_test)
    scores.append(score)

k_optimal=scores.index(max(scores))+1


model_knn = neighbors.KNeighborsClassifier(k_optimal)
model_knn.fit(data_train,label_train)

print("SCORE: ", model_knn.score(data_test,label_test))

predicted_label_test = model_knn.predict(data_test)

counter_wrong_predictions=0
for counter in range(len(label_test)):
    if(counter_wrong_predictions<2):
        if (predicted_label_test[counter] != label_test[counter]):
            counter_wrong_predictions += 1
            print("PREDICTED: ", predicted_label_test[counter])
            print("REAL: ", label_test[counter])
            pl.gray()
            pl.matshow(data_test[counter].reshape(8,8))
            pl.show()


