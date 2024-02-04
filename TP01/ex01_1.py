from sklearn.datasets import load_iris

irisData = load_iris()

print(len(irisData.data))
print(len(irisData.target))
#help(len) # pour quitter l’aide, tapez ’q’ (pour quit)
print(irisData.target_names[0])
print(irisData.target_names[2])
print(irisData.target_names[-1])
print(irisData.target_names[len(irisData.target_names)-1])
print(irisData.data.shape)
print(irisData.data[0])
print(irisData.data[0][1])
print(irisData.data[:,1])