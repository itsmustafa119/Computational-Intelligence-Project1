import numpy as np


class Perceptron:
    def __init__(self, learning_rate = 0.1):
        self.weights = np.array([0.3, 0.5, 0.2])
        self.learning_rate = learning_rate

    def step(self, z):
        if z >= 0:
            return 1
        else:
            return 0
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict_step(self, x1, x2):
        X = np.array([x1, x2, -1])

        z = np.dot(X, self.weights)

        return self.step(z)
    
    def predict_sigmoid(self, x1, x2):
        X = np.array([x1, x2, -1])

        z = np.dot(X, self.weights)

        return self.sigmoid(z)

    def error_correction(self, X_train, y_train, epochs=100):

        error_history = []
        for epoch in range(epochs):
            for x, y in zip(X_train, y_train):    
                output = self.predict_step(x[0], x[1])
                error = y - output


                X = np.array([x[0], x[1], -1])


                self.weights += self.learning_rate * error * X

            mistakes = 0

            for x, y in zip(X_train, y_train):
                output = self.predict_step(x[0], x[1])

                if output != y:
                    mistakes += 1

            error_history.append(mistakes)

        return error_history
                 

    def generalized_delta(self, X_train, y_train, epochs=100):

        error_history = []
        for epoch in range(epochs):
            for x, y in zip(X_train, y_train):
                output = self.predict_sigmoid(x[0], x[1])
                X = np.array([x[0], x[1], -1])
                error = y - output
                delta = error * output * (1 - output)
                self.weights += self.learning_rate * delta * X

            total_error = 0

            for x, y in zip(X_train, y_train):
                output = self.predict_sigmoid(x[0], x[1])

                total_error += 0.5 * (y - output) ** 2

            error_history.append(total_error)

        return error_history
                 

    



