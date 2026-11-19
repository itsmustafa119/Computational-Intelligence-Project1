
import os
import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset2 import Dataset2, TEST_SIZE, SPLIT_SEED

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mlp import CIPerceptron_main


# =====================================
# 1. Load Dataset 2
# =====================================

dataset = Dataset2()

X, y = dataset.create()

X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=int)

print("Dataset shape:", X.shape)
print("Labels:", np.unique(y))


# =====================================
# 2. Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SPLIT_SEED,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================================
# 3. Create MLP
# =====================================

model = CIPerceptron_main(
    input_size=2,
    hidden_size=4,
    output_size=1,
    learning_rate=0.1,
    seed=5
)


# =====================================
# 4. Train Network
# =====================================

print("\nTraining started...")

history = model.fit_ravand(
    X_train,
    y_train,
    epochs=1000
)

print("\nTraining completed!")


# =====================================
# 5. Test Network
# =====================================

y_pred = model.predict(X_test).ravel()

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("FINAL RESULTS")
print("================================")

print("Final Training Loss:", history[-1])

print("Test Accuracy:", accuracy)

print("Test Accuracy (%):", accuracy * 100)


# =====================================
# 6. Create Output Folder
# =====================================

os.makedirs("outputs", exist_ok=True)


# =====================================
# 7. Plot Training Loss
# =====================================

plt.figure(figsize=(8, 5))

plt.plot(history)

plt.xlabel("Epoch")
plt.ylabel("Mean SSE")

plt.title("MLP Training Loss - Dataset 2")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/stage3_loss.png",
    dpi=300
)

plt.show()


# =====================================
# 8. Decision Boundary
# =====================================

x_min = X[:, 0].min() - 0.2
x_max = X[:, 0].max() + 0.2

y_min = X[:, 1].min() - 0.2
y_max = X[:, 1].max() + 0.2

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(grid)

Z = Z.reshape(xx.shape)


plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap="coolwarm"
)

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="coolwarm",
    edgecolors="black",
    label="Training data"
)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap="coolwarm",
    marker="x",
    label="Test data"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("MLP Decision Boundary - Dataset 2")

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/stage3_decision_boundary.png",
    dpi=300
)

plt.show()


# =====================================
# 9. Save Model Settings
# =====================================

with open("outputs/stage3_results.txt", "w") as file:

    file.write("MLP Stage 3 Results\n")

    file.write("------------------\n")

    file.write(f"Input neurons: {model.input_size}\n")
    file.write(f"Hidden neurons: {model.hidden_size}\n")
    file.write(f"Output neurons: {model.output_size}\n")

    file.write(f"Learning rate: {model.learning_rate}\n")

    file.write("Activation: Sigmoid\n")
    file.write("Loss function: SSE\n")
    file.write("Optimizer: SGD\n")

    file.write(f"Epochs: {len(history)}\n")

    file.write(f"Final training loss: {history[-1]}\n")

    file.write(f"Test accuracy: {accuracy}\n")
    file.write("\nDataset Settings\n")
    file.write("------------------\n")
    file.write("Dataset: make_circles\n")
    file.write("Samples: 300\n")
    file.write("Noise: 0.05\n")
    file.write("Factor: 0.5\n")
    file.write("Dataset random_state: 119\n")
    file.write("Train-test ratio: 80/20\n")
    file.write("Split random_state: 7349\n")
    file.write("Stratify: True\n")

    file.write(f"Initialization seed: {model.seed}\n")

print("\nResults saved in outputs folder.")
