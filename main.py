from datasets.Dataset1 import Dataset1
from datasets.dataset2 import Dataset2
from perceptron import Perceptron

perceptron = Perceptron()
dataset1 = Dataset1()
X, y = dataset1.create()
perceptron.error_correction(X, y)
print("dataset 1 error_correction")
print(perceptron.weights)
perceptron.generalized_delta(X, y)
print("dataset 1 generalized delta")
print(perceptron.weights)

dataset2 = Dataset2()
X, y = dataset2.create()
perceptron.error_correction(X, y)
print("dataset 2 error_correction")
print(perceptron.weights)
perceptron.generalized_delta(X, y)
print("dataset 2 generalized delta")
print(perceptron.weights)



