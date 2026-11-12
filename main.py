from datasets.Dataset1 import Dataset1
from perceptron import Perceptron

perceptron = Perceptron()
dataset1 = Dataset1()
X, y = dataset1.create()
perceptron.error_correction(X, y)
print(perceptron.weights)




