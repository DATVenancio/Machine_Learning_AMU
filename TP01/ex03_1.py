from sklearn.datasets import load_digits
import pylab as pl
digits=load_digits()
print(digits.data[0])

print(digits.images[0])
print(digits.data[0].reshape(8,8))
print(digits.target[0])
print(digits.images[0])
print(digits.data[0].reshape(8,8))
print(digits.target[0])

pl.gray()
pl.matshow(digits.images[3])
pl.show()