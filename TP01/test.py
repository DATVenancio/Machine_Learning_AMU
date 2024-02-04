from sklearn.model_selection import KFold
kf=KFold(n_splits=5,shuffle=True)
X=[i for i in range(20)]
print(X)
for learn,test in kf.split(X):
    print("learn: ",learn)
    print("test: ",test)