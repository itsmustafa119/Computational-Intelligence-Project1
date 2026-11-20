
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================================================
# 1. Generate Dataset 2
# =========================================================

X, y = make_circles(
    n_samples=300,
    factor=0.5,
    noise=0.10,
    random_state=42
)

# Change labels to column vector
y = y.reshape(-1, 1)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=7349,
    stratify=y
)


# =========================================================
# 2. Main Neural Network Class
# =========================================================

class CIPerceptron_main:

    def __init__(
        self,
        input_size=2,
        hidden_size=20,
        output_size=1,
        learning_rate=0.1,
        epochs=3000,
        seed=1
    ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

        self.initialize_weights()

    # -----------------------------------------------------
    # Sigmoid activation function
    # -----------------------------------------------------

    def sigmoid(self, z):

        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    # -----------------------------------------------------
    # Initialize weights
    # -----------------------------------------------------

    def initialize_weights(self):

        rng = np.random.default_rng(self.seed)

        self.W1 = rng.uniform(
            -0.5,
            0.5,
            (self.input_size, self.hidden_size)
        )

        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = rng.uniform(
            -0.5,
            0.5,
            (self.hidden_size, self.output_size)
        )

        self.b2 = np.zeros((1, self.output_size))

    # -----------------------------------------------------
    # Feed Forward
    # -----------------------------------------------------

    def feedforward(self, X):

        self.z1 = X @ self.W1 + self.b1

        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2

        self.a2 = self.sigmoid(self.z2)

        return self.a2

    # -----------------------------------------------------
    # Calculate SSE loss
    # -----------------------------------------------------

    def calculate_loss(self, y_true, y_pred):

        return np.mean(
            np.sum((y_true - y_pred) ** 2, axis=1)
        ) / 2

    # -----------------------------------------------------
    # Backpropagation and weight update
    # -----------------------------------------------------

    def backpropagation(self, X, y_true):

        n = X.shape[0]

        # Output layer error
        delta_output = (
            (self.a2 - y_true)
            * self.a2
            * (1 - self.a2)
        )

        # Hidden layer error
        delta_hidden = (
            delta_output @ self.W2.T
        ) * self.a1 * (1 - self.a1)

        # Calculate gradients
        dW2 = (self.a1.T @ delta_output) / n
        db2 = np.mean(delta_output, axis=0, keepdims=True)

        dW1 = (X.T @ delta_hidden) / n
        db1 = np.mean(delta_hidden, axis=0, keepdims=True)

        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # -----------------------------------------------------
    # Training function
    # -----------------------------------------------------

    def fit_ravand(self, X, y):

        self.loss_history = []

        for epoch in range(self.epochs):

            # Forward pass
            predictions = self.feedforward(X)

            # Calculate loss
            loss = self.calculate_loss(y, predictions)

            self.loss_history.append(loss)

            # Backpropagation
            self.backpropagation(X, y)

        return self.loss_history

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    def predict_proba(self, X):

        return self.feedforward(X)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)


# =========================================================
# 3. Train 13 Neural Networks with Different Seeds
# =========================================================

number_of_networks = 13

networks = []

individual_accuracies = []

print("=" * 55)
print("TRAINING 13 NEURAL NETWORKS")
print("=" * 55)

for seed in range(1, number_of_networks + 1):

    model = CIPerceptron_main(
        input_size=2,
        hidden_size=20,
        output_size=1,
        learning_rate=0.1,
        epochs=3000,
        seed=seed
    )

    # Train model
    model.fit_ravand(X_train, y_train)

    # Test prediction
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    networks.append(model)

    individual_accuracies.append(accuracy)

    print(
        f"Network {seed:02d} | "
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )


# =========================================================
# 4. Calculate Mean and Standard Deviation
# =========================================================

mean_accuracy = np.mean(individual_accuracies)

std_accuracy = np.std(individual_accuracies)

print("\n" + "=" * 55)
print("INDIVIDUAL NETWORK RESULTS")
print("=" * 55)

print(
    f"Mean Test Accuracy: "
    f"{mean_accuracy * 100:.2f}%"
)

print(
    f"Standard Deviation: "
    f"{std_accuracy * 100:.2f}%"
)


# =========================================================
# 5. Ensemble: Average the Outputs of 13 Networks
# =========================================================

all_probabilities = []

for model in networks:

    probabilities = model.predict_proba(X_test)

    all_probabilities.append(probabilities)


# Convert predictions to NumPy array
all_probabilities = np.array(all_probabilities)

# Average probabilities of all networks
ensemble_probabilities = np.mean(
    all_probabilities,
    axis=0
)

# Final ensemble classification
ensemble_predictions = (
    ensemble_probabilities >= 0.5
).astype(int)


# Calculate ensemble accuracy
ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_predictions
)

print("\n" + "=" * 55)
print("ENSEMBLE RESULTS")
print("=" * 55)

print(
    f"Ensemble Test Accuracy: "
    f"{ensemble_accuracy * 100:.2f}%"
)

print("\nEnsemble Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        ensemble_predictions
    )
)


# =========================================================
# 6. Compare Individual and Ensemble Accuracy
# =========================================================

print("\n" + "=" * 55)
print("ACCURACY COMPARISON")
print("=" * 55)

print(
    f"Mean Individual Accuracy: "
    f"{mean_accuracy * 100:.2f}%"
)

print(
    f"Ensemble Accuracy: "
    f"{ensemble_accuracy * 100:.2f}%"
)

print(
    f"Accuracy Difference: "
    f"{(ensemble_accuracy - mean_accuracy) * 100:.2f}%"
)


# =========================================================
# 7. Plot Accuracy of 13 Networks
# =========================================================

plt.figure(figsize=(9, 5))

network_numbers = np.arange(
    1,
    number_of_networks + 1
)

plt.bar(
    network_numbers,
    np.array(individual_accuracies) * 100
)

plt.axhline(
    mean_accuracy * 100,
    linestyle="--",
    label="Mean Individual Accuracy"
)

plt.axhline(
    ensemble_accuracy * 100,
    linestyle=":",
    label="Ensemble Accuracy"
)

plt.xlabel("Network Number")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy of 13 Neural Networks")

plt.xticks(network_numbers)
plt.ylim(0, 105)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# 8. Plot Training Loss Curves
# =========================================================

plt.figure(figsize=(10, 6))

for i, model in enumerate(networks):

    plt.plot(
        model.loss_history,
        alpha=0.65,
        label=f"Seed {i + 1}"
    )

plt.xlabel("Epoch")
plt.ylabel("Training SSE Loss")
plt.title("Training Loss Curves of 13 Networks")

plt.legend(ncol=2)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# 9. Prepare Mesh Grid for Decision Boundaries
# =========================================================

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


# =========================================================
# 10. Plot Decision Boundaries of Individual Networks
# =========================================================

selected_indices = [0, 1, 2, 3]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

for ax, index in zip(
    axes.ravel(),
    selected_indices
):

    model = networks[index]

    grid_predictions = model.predict(grid)

    grid_predictions = grid_predictions.reshape(
        xx.shape
    )

    ax.contourf(
        xx,
        yy,
        grid_predictions,
        alpha=0.3,
        levels=[-0.5, 0.5, 1.5]
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y.ravel(),
        edgecolors="black",
        s=25
    )

    ax.set_title(
        f"Individual Network - Seed {index + 1}"
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.suptitle(
    "Decision Boundaries of Individual Networks"
)

plt.tight_layout()
plt.show()


# =========================================================
# 11. Plot Ensemble Decision Boundary
# =========================================================

ensemble_grid_probabilities = []

for model in networks:

    probabilities = model.predict_proba(grid)

    ensemble_grid_probabilities.append(probabilities)


ensemble_grid_probabilities = np.array(
    ensemble_grid_probabilities
)

# Average output of 13 networks
ensemble_grid = np.mean(
    ensemble_grid_probabilities,
    axis=0
)

ensemble_grid = (
    ensemble_grid >= 0.5
).astype(int)

ensemble_grid = ensemble_grid.reshape(
    xx.shape
)


plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    ensemble_grid,
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

plt.title(
    "Ensemble Decision Boundary (13 Networks)"
)

plt.tight_layout()
plt.show()


# =========================================================
# 12. Compare One Individual Network with Ensemble
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

# Individual network
individual_grid = networks[0].predict(grid)

individual_grid = individual_grid.reshape(
    xx.shape
)

axes[0].contourf(
    xx,
    yy,
    individual_grid,
    alpha=0.3,
    levels=[-0.5, 0.5, 1.5]
)

axes[0].scatter(
    X[:, 0],
    X[:, 1],
    c=y.ravel(),
    edgecolors="black",
    s=25
)

axes[0].set_title("Individual Network - Seed 1")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")


# Ensemble network
axes[1].contourf(
    xx,
    yy,
    ensemble_grid,
    alpha=0.3,
    levels=[-0.5, 0.5, 1.5]
)

axes[1].scatter(
    X[:, 0],
    X[:, 1],
    c=y.ravel(),
    edgecolors="black",
    s=25
)

axes[1].set_title("Ensemble - 13 Networks")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")


plt.suptitle(
    "Individual Network vs Ensemble"
)

plt.tight_layout()
plt.show()


# =========================================================
# 13. Final Summary
# =========================================================

print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)

print(f"Number of Networks: {number_of_networks}")
print(f"Hidden Neurons: 20")

print(
    f"Mean Individual Accuracy: "
    f"{mean_accuracy * 100:.2f}%"
)

print(
    f"Standard Deviation: "
    f"{std_accuracy * 100:.2f}%"
)

print(
    f"Ensemble Accuracy: "
    f"{ensemble_accuracy * 100:.2f}%"
)

print("\nStage 6 completed successfully.")
