import numpy as np


class Perceptron:
    def __init__(self):
        self.weights = np.array([0.3, 0.5, 0.2])

    def step(self, z):
        if z >= 0:
            return 1
        else:
            return 0

    def predict(self, x1, x2):
        X = np.array([x1, x2, -1])

        z = np.dot(X, self.weights)

        return self.step(z)



