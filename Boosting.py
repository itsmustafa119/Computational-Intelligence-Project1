
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# =====================================================
# 1. Generate Dataset 2
# =====================================================

X, y = make_circles(
    n_samples=300,
    factor=0.5,
    noise=0.10,
    random_state=42
)

y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=7349,
    stratify=y
)


# =====================================================
# 2. Neural Network Class
# =====================================================

class CIPerceptron_main:

    def __init__(
        self,
        input_size=2,
        hidden_size=20,
        output_size=1,
        learning_rate=0.1,
        epochs=2000,
        seed=1
    ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

        self.initialize_weights()

    # Sigmoid
    def sigmoid(self, z):

        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    # Initialize weights
    def initialize_weights(self):

        rng = np.random.default_rng(self.seed)

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

    # Forward propagation
    def feedforward(self, X):

        self.z1 = X @ self.W1 + self.b1

        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2

        self.a2 = self.sigmoid(self.z2)

        return self.a2

    # Backpropagation
    def backpropagation(self, X, y):

        n = X.shape[0]

        delta_output = (
            (self.a2 - y)
            * self.a2
            * (1 - self.a2)
        )

        delta_hidden = (
            delta_output @ self.W2.T
        ) * self.a1 * (1 - self.a1)

        dW2 = self.a1.T @ delta_output / n
        db2 = np.mean(delta_output, axis=0, keepdims=True)

        dW1 = X.T @ delta_hidden / n
        db1 = np.mean(delta_hidden, axis=0, keepdims=True)

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # Training
    def fit_ravand(self, X, y):

        self.loss_history = []

        for epoch in range(self.epochs):

            predictions = self.feedforward(X)

            loss = np.mean(
                np.sum((y - predictions) ** 2, axis=1)
            ) / 2

            self.loss_history.append(loss)

            self.backpropagation(X, y)

        return self.loss_history

    # Probability prediction
    def predict_proba(self, X):

        return self.feedforward(X)

    # Class prediction
    def predict(self, X):

        return (
            self.predict_proba(X) >= 0.5
        ).astype(int)


# =====================================================
# 3. Boosting Algorithm
# =====================================================

class SimpleBoosting:

    def __init__(
        self,
        n_estimators=13,
        hidden_size=20,
        learning_rate=0.1,
        epochs=2000
    ):

        self.n_estimators = n_estimators
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.models = []
        self.alphas = []
        self.errors = []

    def fit(self, X, y):

        n_samples = len(X)

        # Initially all samples have equal weights
        sample_weights = np.ones(n_samples) / n_samples

        self.models = []
        self.alphas = []
        self.errors = []

        for round_number in range(self.n_estimators):

            print(
                f"\nTraining Boosting Network "
                f"{round_number + 1}"
            )

            # -----------------------------------------
            # Weighted sampling
            # -----------------------------------------

            rng = np.random.default_rng(
                100 + round_number
            )

            indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights
            )

            X_sample = X[indices]
            y_sample = y[indices]

            # -----------------------------------------
            # Create and train a new neural network
            # -----------------------------------------

            model = CIPerceptron_main(
                input_size=2,
                hidden_size=self.hidden_size,
                output_size=1,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                seed=round_number + 1
            )

            model.fit_ravand(
                X_sample,
                y_sample
            )

            # -----------------------------------------
            # Evaluate on original training data
            # -----------------------------------------

            predictions = model.predict(X).ravel()

            true_labels = y.ravel()

            incorrect = (
                predictions != true_labels
            )

            # Weighted classification error
            error = np.sum(
                sample_weights * incorrect
            )

            print(
                f"Weighted Error: {error:.4f}"
            )

            # -----------------------------------------
            # Weak learner condition
            # -----------------------------------------

            if error >= 0.5:

                print(
                    "This network is not better "
                    "than random guessing. Stopping."
                )

                break

            # Avoid division by zero
            error = np.clip(
                error,
                1e-10,
                1 - 1e-10
            )

            # -----------------------------------------
            # Calculate model weight
            # -----------------------------------------

            alpha = 0.5 * np.log(
                (1 - error) / error
            )

            self.models.append(model)
            self.alphas.append(alpha)
            self.errors.append(error)

            print(
                f"Network Weight (alpha): {alpha:.4f}"
            )

            # -----------------------------------------
            # Update sample weights
            # -----------------------------------------

            sample_weights *= np.exp(
                alpha * incorrect
            )

            # Normalize sample weights
            sample_weights /= np.sum(sample_weights)

            print(
                "Maximum sample weight:",
                np.max(sample_weights)
            )

            # If the model classifies all training
            # samples correctly, no further round
            # is necessary.
            if error <= 1e-10:
                print("Perfect training classification.")
                break

        return self

    # =================================================
    # Weighted Voting
    # =================================================

    def predict(self, X):

        if len(self.models) == 0:
            raise RuntimeError(
                "No valid boosting model was trained."
            )

        weighted_vote = np.zeros(len(X))

        for model, alpha in zip(
            self.models,
            self.alphas
        ):

            predictions = model.predict(X).ravel()

            # Convert labels 0 and 1 to -1 and +1
            signed_predictions = (
                2 * predictions - 1
            )

            weighted_vote += (
                alpha * signed_predictions
            )

        return (
            weighted_vote >= 0
        ).astype(int)


# =====================================================
# 4. Train Boosting
# =====================================================

boosting_model = SimpleBoosting(
    n_estimators=13,
    hidden_size=20,
    learning_rate=0.1,
    epochs=2000
)

boosting_model.fit(
    X_train,
    y_train
)


# =====================================================
# 5. Evaluate Boosting
# =====================================================

boosting_predictions = boosting_model.predict(
    X_test
)

boosting_accuracy = accuracy_score(
    y_test.ravel(),
    boosting_predictions
)

print("\n===================================")
print("BOOSTING RESULTS")
print("===================================")

print(
    "Number of trained networks:",
    len(boosting_model.models)
)

print(
    f"Boosting Test Accuracy: "
    f"{boosting_accuracy * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test.ravel(),
        boosting_predictions
    )
)


# =====================================================
# 6. Plot Weighted Errors
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(boosting_model.errors) + 1),
    boosting_model.errors,
    marker="o"
)

plt.xlabel("Boosting Round")
plt.ylabel("Weighted Classification Error")

plt.title("Boosting: Weighted Error per Network")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# 7. Plot Model Weights
# =====================================================

plt.figure(figsize=(8, 5))

plt.bar(
    range(1, len(boosting_model.alphas) + 1),
    boosting_model.alphas
)

plt.xlabel("Network Number")
plt.ylabel("Alpha (Model Weight)")

plt.title("Importance of Each Boosting Network")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# 8. Decision Boundary
# =====================================================

x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5

y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250)
)

grid = np.c_[
    xx.ravel(),
    yy.ravel()
]

grid_predictions = boosting_model.predict(
    grid
).reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    grid_predictions,
    alpha=0.3,
    levels=[-0.5, 0.5, 1.5]
)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y.ravel(),
    edgecolors="black",
    s=30
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("Boosting Decision Boundary")

plt.tight_layout()
plt.show()


print("\nBoosting completed successfully.")
