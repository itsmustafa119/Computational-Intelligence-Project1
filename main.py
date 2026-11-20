import matplotlib.pyplot as plt

from datasets.Dataset1 import Dataset1
from datasets.dataset2 import Dataset2
from perceptron import Perceptron

perceptron = Perceptron()
dataset1 = Dataset1()
X, y = dataset1.create()
step_errors_1 = perceptron.error_correction(X, y)
print("dataset 1 error_correction")
print(perceptron.weights)
sigmoid_errors_1 = perceptron.generalized_delta(X, y)
print("dataset 1 generalized delta")
print(perceptron.weights)

dataset2 = Dataset2()
X, y = dataset2.create()
step_errors_2 = perceptron.error_correction(X, y)
print("dataset 2 error_correction")
print(perceptron.weights)
sigmoid_errors_2 = perceptron.generalized_delta(X, y)
print("dataset 2 generalized delta")
print(perceptron.weights)





def plot_error(error_history, title, ylabel):
    plt.figure(figsize=(8, 5))

    plt.plot(range(1, len(error_history) + 1), error_history)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    plt.show()

plot_error(step_errors_1, "Step Perceptron dataset 1", "Number of mistakes")
plot_error(sigmoid_errors_1, "Sigmoid Perceptron dataset 1", "Number of mistakes")
plot_error(step_errors_2, "Step Perceptron dataset 2", "Number of mistakes")
plot_error(sigmoid_errors_2, "Sigmoid Perceptron dataset 2", "Number of mistakes")

print (len(step_errors_1))