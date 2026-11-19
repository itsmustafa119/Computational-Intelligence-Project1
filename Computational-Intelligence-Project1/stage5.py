
# -*- coding: utf-8 -*-
"""
Computational Intelligence - Project 1 - Stage 5
Three-class MLP implemented from scratch with NumPy.
Compares raw inputs against standardized inputs.

Allowed sklearn usage: dataset generation, train/test split, and metrics.
The neural network training itself is implemented only with NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ------------------------- Reproducible configuration -------------------------

DATA_SEED = 42
SPLIT_SEED = 7349
WEIGHT_SEED = 5

N_SAMPLES = 300
N_CLASSES = 3
HIDDEN_NEURONS = 4
LEARNING_RATE = 0.1
EPOCHS = 1000


# ------------------------------ Dataset 3 ------------------------------------

def create_dataset():
    """
    Dataset 3: three two-dimensional classes generated with make_blobs.
    The centers are deliberately separated to create a clear 3-class problem.
    """
    X, y = make_blobs(
        n_samples=N_SAMPLES,
        centers=3,
        n_features=2,
        cluster_std=1.5,
        random_state=DATA_SEED
    )
    return X.astype(float), y.astype(int)


# ------------------------- MLP implemented in NumPy --------------------------

class CIPerceptron_main:
    """
    One-hidden-layer MLP:
        2 input neurons -> 4 sigmoid hidden neurons -> 3 sigmoid output neurons

    The output uses one-hot targets and argmax for final classification.
    Loss: Sum of Squared Errors (SSE), averaged over samples for plotting.
    """

    def __init__(self, input_size=2, hidden_size=4, output_size=3,
                 learning_rate=0.1, epochs=1000, seed=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.loss_history = []
        self.initialize_weights()

    @staticmethod
    def sigmoid(z):
        # Clipping prevents numerical overflow in exp.
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def initialize_weights(self):
        rng = np.random.default_rng(self.seed)
        # Same initialization convention as Stage 3: uniform [-0.5, 0.5].
        self.W1 = rng.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = rng.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))
        self.b2 = np.zeros((1, self.output_size))

    def feedforward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backpropagation(self, X, y_onehot):
        """
        Gradient of half-SSE = 0.5 * sum((target-output)^2).
        This is equivalent to minimizing SSE; the constant 0.5 does not
        change the minimizer and keeps the derivative compact.
        """
        output = self.feedforward(X)

        delta2 = (output - y_onehot) * output * (1.0 - output)
        grad_W2 = self.a1.T @ delta2 / X.shape[0]
        grad_b2 = np.mean(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ self.W2.T) * self.a1 * (1.0 - self.a1)
        grad_W1 = X.T @ delta1 / X.shape[0]
        grad_b1 = np.mean(delta1, axis=0, keepdims=True)

        return grad_W1, grad_b1, grad_W2, grad_b2

    def update_weights(self, gradients):
        grad_W1, grad_b1, grad_W2, grad_b2 = gradients
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2

    def fit_ravand(self, X, y):
        y_onehot = np.eye(self.output_size)[y]
        self.loss_history = []

        for epoch in range(self.epochs):
            gradients = self.backpropagation(X, y_onehot)
            self.update_weights(gradients)

            # Record mean SSE after the update.
            predictions = self.feedforward(X)
            loss = np.mean(np.sum((y_onehot - predictions) ** 2, axis=1))
            self.loss_history.append(float(loss))

        return self

    def predict_proba(self, X):
        return self.feedforward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ----------------------------- Standardization -------------------------------

def standardize_using_train_only(X_train, X_test):
    """
    Fit mean/std on training data only, then transform both train and test.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled, mean, std


# --------------------------------- Plotting ----------------------------------

def plot_dataset(X, y):
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis",
                          edgecolors="k", alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Dataset 3 - Three Classes")
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y, title, resolution=250):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.28, cmap="viridis",
                 levels=np.arange(N_CLASSES + 1) - 0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis",
                edgecolors="k", s=24)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()


def plot_loss_comparison(raw_model, normalized_model):
    plt.figure(figsize=(8, 5))
    plt.plot(raw_model.loss_history, label="Raw input")
    plt.plot(normalized_model.loss_history, label="Standardized input")
    plt.xlabel("Epoch")
    plt.ylabel("Mean SSE")
    plt.title("Training Loss: Raw vs Standardized Inputs")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


# ---------------------------------- Main --------------------------------------

def run_experiment():
    X, y = create_dataset()

    # One fixed stratified split, shared by both experiments.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SPLIT_SEED,
        stratify=y
    )

    print("Dataset shape:", X.shape)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Dataset generation: make_blobs(n_samples=300, centers=3, "
          "n_features=2, cluster_std=1.5, random_state=42)")
    print("Split: 80/20, stratified, random_state=7349")

    plot_dataset(X, y)

    # Experiment 1: raw features.
    raw_model = CIPerceptron_main(
        input_size=2,
        hidden_size=HIDDEN_NEURONS,
        output_size=N_CLASSES,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        seed=WEIGHT_SEED
    )
    raw_model.fit_ravand(X_train, y_train)
    raw_predictions = raw_model.predict(X_test)
    raw_accuracy = accuracy_score(y_test, raw_predictions)
    raw_cm = confusion_matrix(y_test, raw_predictions, labels=[0, 1, 2])

    # Experiment 2: standardized features; parameters fitted only on train.
    X_train_scaled, X_test_scaled, train_mean, train_std = (
        standardize_using_train_only(X_train, X_test)
    )

    normalized_model = CIPerceptron_main(
        input_size=2,
        hidden_size=HIDDEN_NEURONS,
        output_size=N_CLASSES,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        seed=WEIGHT_SEED
    )
    normalized_model.fit_ravand(X_train_scaled, y_train)
    normalized_predictions = normalized_model.predict(X_test_scaled)
    normalized_accuracy = accuracy_score(y_test, normalized_predictions)
    normalized_cm = confusion_matrix(
        y_test, normalized_predictions, labels=[0, 1, 2]
    )

    # Report metrics.
    print("\n" + "=" * 60)
    print("RESULTS - RAW INPUT")
    print("=" * 60)
    print(f"Test accuracy: {raw_accuracy:.4f} ({raw_accuracy * 100:.2f}%)")
    print("Confusion matrix (rows=true class, columns=predicted class):")
    print(raw_cm)
    print(f"Final training loss: {raw_model.loss_history[-1]:.6f}")

    print("\n" + "=" * 60)
    print("RESULTS - STANDARDIZED INPUT")
    print("=" * 60)
    print(f"Test accuracy: {normalized_accuracy:.4f} "
          f"({normalized_accuracy * 100:.2f}%)")
    print("Confusion matrix (rows=true class, columns=predicted class):")
    print(normalized_cm)
    print(f"Final training loss: {normalized_model.loss_history[-1]:.6f}")

    print("\nStandardization parameters (computed from training set only):")
    print("Mean:", train_mean)
    print("Std :", train_std)

    # Visualize confusion matrices.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(raw_cm, display_labels=[0, 1, 2]).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title(f"Raw Input (Accuracy={raw_accuracy:.3f})")
    ConfusionMatrixDisplay(normalized_cm, display_labels=[0, 1, 2]).plot(
        ax=axes[1], cmap="Blues", colorbar=False
    )
    axes[1].set_title(
        f"Standardized Input (Accuracy={normalized_accuracy:.3f})"
    )
    plt.tight_layout()
    plt.show()

    # Decision boundaries: both models are evaluated in their own input space.
    plot_decision_boundary(
        raw_model, X, y, "Three-Class Decision Boundary - Raw Input"
    )
    # For the normalized model, use a wrapper that scales every grid point.
    class ScaledModel:
        def __init__(self, base_model, mean, std):
            self.base_model = base_model
            self.mean = mean
            self.std = std

        def predict(self, X_points):
            return self.base_model.predict((X_points - self.mean) / self.std)

    scaled_model_for_plot = ScaledModel(
        normalized_model, train_mean, train_std
    )
    plot_decision_boundary(
        scaled_model_for_plot, X, y,
        "Three-Class Decision Boundary - Standardized Input"
    )

    plot_loss_comparison(raw_model, normalized_model)

    print("\nAccuracy comparison:")
    print(f"Raw input         : {raw_accuracy * 100:.2f}%")
    print(f"Standardized input: {normalized_accuracy * 100:.2f}%")

    return {
        "raw_model": raw_model,
        "normalized_model": normalized_model,
        "raw_accuracy": raw_accuracy,
        "normalized_accuracy": normalized_accuracy,
        "raw_confusion_matrix": raw_cm,
        "normalized_confusion_matrix": normalized_cm,
        "train_mean": train_mean,
        "train_std": train_std,
    }


if __name__ == "__main__":
    results = run_experiment()
