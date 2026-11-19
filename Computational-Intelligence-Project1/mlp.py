
import numpy as np


class CIPerceptron_main:

    def __init__(
        self,
        input_size=2,
        hidden_size=1,
        output_size=1,
        learning_rate=0.1,
        seed=5
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.seed = seed

        self.initialize_weights(seed)

    # --------------------------------
    # 1. Weight Initialization
    # --------------------------------
    def initialize_weights(self, seed=5):

        rng = np.random.default_rng(seed)

        self.W1 = rng.uniform(
            -0.5, 0.5,
            (self.input_size, self.hidden_size)
        )

        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = rng.uniform(
            -0.5, 0.5,
            (self.hidden_size, self.output_size)
        )

        self.b2 = np.zeros((1, self.output_size))

    # --------------------------------
    # Sigmoid Activation
    # --------------------------------
    def sigmoid(self, z):

        z = np.clip(z, -500, 500)

        return 1.0 / (1.0 + np.exp(-z))

    # --------------------------------
    # 2. Feedforward
    # --------------------------------
    def feedforward(self, X):

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Hidden layer
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.sigmoid(self.Z1)

        # Output layer
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    # --------------------------------
    # 3. Backpropagation
    # --------------------------------
    def backpropagation(self, X, y):

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        y = np.asarray(y, dtype=float)
        y = y.reshape(-1, self.output_size)

        n = X.shape[0]

        # Forward pass
        output = self.feedforward(X)

        # Output error signal
        delta2 = (
            2 * (output - y)
            * output * (1 - output)
        )

        # Hidden error signal
        delta1 = (
            delta2 @ self.W2.T
            * self.A1 * (1 - self.A1)
        )

        # Gradients
        dW2 = self.A1.T @ delta2 / n
        db2 = np.sum(delta2, axis=0, keepdims=True) / n

        dW1 = X.T @ delta1 / n
        db1 = np.sum(delta1, axis=0, keepdims=True) / n

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

    # --------------------------------
    # 4. Update Weights
    # --------------------------------
    def update_weights(self, gradients):

        lr = self.learning_rate

        self.W1 -= lr * gradients["dW1"]
        self.b1 -= lr * gradients["db1"]

        self.W2 -= lr * gradients["dW2"]
        self.b2 -= lr * gradients["db2"]

    # --------------------------------
    # SSE Loss
    # --------------------------------
    def calculate_loss(self, X, y):

        y = np.asarray(y, dtype=float)
        y = y.reshape(-1, self.output_size)

        prediction = self.feedforward(X)

        return np.sum((prediction - y) ** 2)

    # --------------------------------
    # Training Function
    # --------------------------------
    def fit_ravand(self, X, y, epochs=1000):

        X = np.asarray(X, dtype=float)

        y = np.asarray(y, dtype=float)
        y = y.reshape(-1, self.output_size)

        self.loss_history = []

        rng = np.random.default_rng(self.seed)

        for epoch in range(epochs):

            indices = rng.permutation(len(X))

            for i in indices:

                xi = X[i:i+1]
                yi = y[i:i+1]

                gradients = self.backpropagation(xi, yi)

                self.update_weights(gradients)

            # Mean SSE per training sample
            epoch_loss = np.mean([
                self.calculate_loss(
                    X[i:i+1],
                    y[i:i+1]
                )
                for i in range(len(X))
            ])

            self.loss_history.append(epoch_loss)

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Loss: {epoch_loss:.6f}"
                )

        return self.loss_history

    # --------------------------------
    # Prediction
    # --------------------------------
    def predict_proba(self, X):

        return self.feedforward(X)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)
