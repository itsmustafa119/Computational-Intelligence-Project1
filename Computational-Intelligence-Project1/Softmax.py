
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# =====================================================
# 1. Generate Dataset 3
# =====================================================

X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.5,
    random_state=42
)

# One-hot encoding
number_of_classes = 3

Y = np.eye(number_of_classes)[y]


# =====================================================
# 2. Train-Test Split
# =====================================================

X_train, X_test, Y_train, Y_test, y_train, y_test = (
    train_test_split(
        X,
        Y,
        y,
        test_size=0.20,
        random_state=7349,
        stratify=y
    )
)


# =====================================================
# 3. Standardization
# =====================================================

# Calculate mean and standard deviation
# ONLY from training data

mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0)

std[std == 0] = 1

X_train_std = (X_train - mean) / std

X_test_std = (X_test - mean) / std


# =====================================================
# 4. Neural Network Class
# =====================================================

class CIPerceptron_main:

    def __init__(
        self,
        input_size=2,
        hidden_size=4,
        output_size=3,
        learning_rate=0.1,
        epochs=1500,
        seed=5,
        output_mode="softmax"
    ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

        self.output_mode = output_mode

        self.initialize_weights()

    # -------------------------------------------------
    # Sigmoid
    # -------------------------------------------------

    def sigmoid(self, z):

        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    # -------------------------------------------------
    # Softmax
    # -------------------------------------------------

    def softmax(self, z):

        z = z - np.max(
            z,
            axis=1,
            keepdims=True
        )

        exp_z = np.exp(z)

        return exp_z / np.sum(
            exp_z,
            axis=1,
            keepdims=True
        )

    # -------------------------------------------------
    # Initialize Weights
    # -------------------------------------------------

    def initialize_weights(self):

        rng = np.random.default_rng(self.seed)

        self.W1 = rng.uniform(
            -0.5, 0.5,
            (self.input_size, self.hidden_size)
        )

        self.b1 = np.zeros(
            (1, self.hidden_size)
        )

        self.W2 = rng.uniform(
            -0.5, 0.5,
            (self.hidden_size, self.output_size)
        )

        self.b2 = np.zeros(
            (1, self.output_size)
        )

    # -------------------------------------------------
    # Feed Forward
    # -------------------------------------------------

    def feedforward(self, X):

        self.z1 = X @ self.W1 + self.b1

        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2

        if self.output_mode == "softmax":

            self.a2 = self.softmax(self.z2)

        else:

            self.a2 = self.sigmoid(self.z2)

        return self.a2

    # -------------------------------------------------
    # Calculate Loss
    # -------------------------------------------------

    def calculate_loss(self, y_true, y_pred):

        if self.output_mode == "softmax":

            # Cross-Entropy
            eps = 1e-12

            y_pred = np.clip(
                y_pred,
                eps,
                1 - eps
            )

            loss = -np.mean(
                np.sum(
                    y_true * np.log(y_pred),
                    axis=1
                )
            )

        else:

            # Sum of Squared Errors
            loss = np.mean(
                np.sum(
                    (y_true - y_pred) ** 2,
                    axis=1
                )
            ) / 2

        return loss

    # -------------------------------------------------
    # Backpropagation
    # -------------------------------------------------

    def backpropagation(self, X, y_true):

        n = X.shape[0]

        if self.output_mode == "softmax":

            # Softmax + Cross-Entropy
            delta_output = self.a2 - y_true

        else:

            # Sigmoid + SSE
            delta_output = (
                (self.a2 - y_true)
                * self.a2
                * (1 - self.a2)
            )

        # Hidden layer gradient
        delta_hidden = (
            delta_output @ self.W2.T
        ) * self.a1 * (1 - self.a1)

        # Gradients for output layer
        dW2 = self.a1.T @ delta_output / n

        db2 = np.mean(
            delta_output,
            axis=0,
            keepdims=True
        )

        # Gradients for hidden layer
        dW1 = X.T @ delta_hidden / n

        db1 = np.mean(
            delta_hidden,
            axis=0,
            keepdims=True
        )

        # Update weights
        self.W2 -= self.learning_rate * dW2

        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1

        self.b1 -= self.learning_rate * db1

    # -------------------------------------------------
    # Training Function
    # -------------------------------------------------

    def fit_ravand(self, X, y):

        self.loss_history = []

        for epoch in range(self.epochs):

            predictions = self.feedforward(X)

            loss = self.calculate_loss(
                y,
                predictions
            )

            self.loss_history.append(loss)

            self.backpropagation(
                X,
                y
            )

        return self.loss_history

    # -------------------------------------------------
    # Predict Probabilities
    # -------------------------------------------------

    def predict_proba(self, X):

        return self.feedforward(X)

    # -------------------------------------------------
    # Predict Class
    # -------------------------------------------------

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return np.argmax(
            probabilities,
            axis=1
        )


# =====================================================
# 5. Experiment Function
# =====================================================

def run_experiment(
    X_train,
    X_test,
    Y_train,
    y_test,
    mode,
    seed=5
):

    model = CIPerceptron_main(
        input_size=2,
        hidden_size=4,
        output_size=3,
        learning_rate=0.1,
        epochs=1500,
        seed=seed,
        output_mode=mode
    )

    model.fit_ravand(
        X_train,
        Y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return model, accuracy, matrix


# =====================================================
# 6. Run Four Experiments
# =====================================================

results = {}

# A. Sigmoid + SSE on raw data
results["SSE_Raw"] = run_experiment(
    X_train,
    X_test,
    Y_train,
    y_test,
    mode="sigmoid"
)

# B. Softmax + Cross-Entropy on raw data
results["CE_Raw"] = run_experiment(
    X_train,
    X_test,
    Y_train,
    y_test,
    mode="softmax"
)

# C. Sigmoid + SSE on standardized data
results["SSE_Standardized"] = run_experiment(
    X_train_std,
    X_test_std,
    Y_train,
    y_test,
    mode="sigmoid"
)

# D. Softmax + Cross-Entropy on standardized data
results["CE_Standardized"] = run_experiment(
    X_train_std,
    X_test_std,
    Y_train,
    y_test,
    mode="softmax"
)


# =====================================================
# 7. Print Results
# =====================================================

print("\n======================================")
print("STAGE 5: LOSS FUNCTION COMPARISON")
print("======================================")

for name, result in results.items():

    model, accuracy, matrix = result

    print("\nMethod:", name)

    print(
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )

    print("Confusion Matrix:")

    print(matrix)


# =====================================================
# 8. Compare Accuracies
# =====================================================

names = list(results.keys())

accuracies = [
    results[name][1] * 100
    for name in names
]

plt.figure(figsize=(10, 5))

plt.bar(
    names,
    accuracies
)

plt.ylabel("Test Accuracy (%)")

plt.xlabel("Method")

plt.title(
    "Comparison of Sigmoid-SSE and Softmax-CrossEntropy"
)

plt.ylim(0, 105)

plt.xticks(rotation=20)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# =====================================================
# 9. Compare Training Loss Curves
# =====================================================

plt.figure(figsize=(10, 6))

for name, result in results.items():

    model = result[0]

    plt.plot(
        model.loss_history,
        label=name
    )

plt.xlabel("Epoch")

plt.ylabel("Training Loss")

plt.title(
    "Training Loss Comparison"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =====================================================
# 10. Decision Boundary for Each Method
# =====================================================

x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1

y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250)
)

grid_raw = np.c_[
    xx.ravel(),
    yy.ravel()
]

grid_std = (grid_raw - mean) / std

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 10)
)

plot_settings = [
    ("SSE_Raw", grid_raw, "Sigmoid + SSE (Raw)"),
    ("CE_Raw", grid_raw, "Softmax + CE (Raw)"),
    (
        "SSE_Standardized",
        grid_std,
        "Sigmoid + SSE (Standardized)"
    ),
    (
        "CE_Standardized",
        grid_std,
        "Softmax + CE (Standardized)"
    )
]

for ax, (key, grid_data, title) in zip(
    axes.ravel(),
    plot_settings
):

    model = results[key][0]

    predictions = model.predict(
        grid_data
    ).reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        predictions,
        alpha=0.3,
        levels=[-0.5, 0.5, 1.5, 2.5]
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="black",
        s=25
    )

    ax.set_title(title)

    ax.set_xlabel("Feature 1")

    ax.set_ylabel("Feature 2")

plt.suptitle(
    "Decision Boundaries: Stage 5 Comparison"
)

plt.tight_layout()
plt.show()


# =====================================================
# 11. Final Comparison
# =====================================================

print("\n======================================")
print("FINAL COMPARISON")
print("======================================")

print(
    "Sigmoid + SSE Raw:",
    f"{results['SSE_Raw'][1] * 100:.2f}%"
)

print(
    "Softmax + CE Raw:",
    f"{results['CE_Raw'][1] * 100:.2f}%"
)

print(
    "Sigmoid + SSE Standardized:",
    f"{results['SSE_Standardized'][1] * 100:.2f}%"
)

print(
    "Softmax + CE Standardized:",
    f"{results['CE_Standardized'][1] * 100:.2f}%"
)

print("\nStage 5 comparison completed.")
برا