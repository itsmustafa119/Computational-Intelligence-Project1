
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mlp import CIPerceptron_main
from datasets.dataset2 import Dataset2


# ==========================================
# GENERAL SETTINGS
# ==========================================

EPOCHS = 1000
SEED = 5
TEST_SIZE = 0.2

INPUT_SIZE = 2
OUTPUT_SIZE = 1

OUTPUT_DIR = "outputs/stage4"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# LOAD DATASET 2
# ==========================================

dataset = Dataset2()
X, y = dataset.create()

X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=7349,
    stratify=y.reshape(-1)
)

print("Dataset 2 loaded successfully")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# CREATE AND INITIALIZE MODEL
# ==========================================

def create_model(hidden_size=4, learning_rate=0.1,
                 initialization="default", seed=SEED,    initialization_seed=None
                ):

    model = CIPerceptron_main(
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        output_size=OUTPUT_SIZE,
        learning_rate=learning_rate,
        seed=seed
    )
     # Separate initialization randomness from training order
    init_seed = (
        seed if initialization_seed is None
        else initialization_seed
    )

    rng = np.random.default_rng(init_seed)

    if initialization == "zero":
        model.W1 = np.zeros_like(model.W1)
        model.W2 = np.zeros_like(model.W2)

    elif initialization == "uniform_5":
        model.W1 = rng.uniform(-5, 5, size=model.W1.shape)
        model.W2 = rng.uniform(-5, 5, size=model.W2.shape)

    elif initialization == "uniform_10":
        model.W1 = rng.uniform(-10, 10, size=model.W1.shape)
        model.W2 = rng.uniform(-10, 10, size=model.W2.shape)

    # Biases are zero in all experiments
    model.b1 = np.zeros_like(model.b1)
    model.b2 = np.zeros_like(model.b2)

    return model


# ==========================================
# TRAIN AND EVALUATE
# ==========================================

def train_and_evaluate(hidden_size=4, learning_rate=0.1,
                       initialization="default", seed=SEED,    initialization_seed=None
                    ):

    model = create_model(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        initialization=initialization,
        seed=seed,
        initialization_seed=initialization_seed

    )

    model.fit_ravand(
        X_train,
        y_train,
        epochs=EPOCHS,
       
    )

    predictions = model.predict(X_test).reshape(-1)

    accuracy = accuracy_score(
        y_test.reshape(-1),
        predictions
    )

    final_loss = model.loss_history[-1]

    return model, final_loss, accuracy


# ==========================================
# SAVE RESULTS TO CSV
# ==========================================

def save_results(filename, results, headers):

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)

    print("Saved:", path)


# ==========================================
# EXPERIMENT 1: WEIGHT INITIALIZATION
# ==========================================

print("\n" + "=" * 55)
print("EXPERIMENT 1: WEIGHT INITIALIZATION")
print("=" * 55)

initialization_results = []
random_small_results = []

# A) Zero initialization: one run

print("\nRunning: All weights zero")

zero_model, loss, accuracy = train_and_evaluate(
    hidden_size=4,
    learning_rate=0.1,
    initialization="zero",
    seed=SEED,
    initialization_seed=SEED
)

initialization_results.append([
    "Zero",
    SEED,
    loss,
    accuracy
])

print("Final Loss:", loss)
print("Test Accuracy:", accuracy)


# B) Small random initialization: five seeds

random_seeds = [5, 21, 42, 73, 100]

print("\nRunning Uniform [-5, 5] with five seeds")

for init_seed in random_seeds:

    model, loss, accuracy = train_and_evaluate(
        hidden_size=4,
        learning_rate=0.1,
        initialization="uniform_5",
        seed=SEED,
        initialization_seed=init_seed
    )

    random_small_results.append([loss, accuracy])

    initialization_results.append([
        "Uniform [-5, 5]",
        init_seed,
        loss,
        accuracy
    ])

    print(
        f"Seed: {init_seed} | "
        f"Loss: {loss:.8f} | "
        f"Accuracy: {accuracy:.4f}"
    )


random_small_results = np.asarray(random_small_results)

mean_loss = np.mean(random_small_results[:, 0])
std_loss = np.std(random_small_results[:, 0])

mean_accuracy = np.mean(random_small_results[:, 1])
std_accuracy = np.std(random_small_results[:, 1])

print("\nUniform [-5, 5] Summary")

print(f"Mean Loss: {mean_loss:.8f}")
print(f"Std Loss: {std_loss:.8f}")

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Std Accuracy: {std_accuracy:.4f}")


# C) Large random initialization: one run

print("\nRunning: Uniform [-10, 10]")

model, loss, accuracy = train_and_evaluate(
    hidden_size=4,
    learning_rate=0.1,
    initialization="uniform_10",
    seed=SEED,
    initialization_seed=SEED
)

initialization_results.append([
    "Uniform [-10, 10]",
    SEED,
    loss,
    accuracy
])

print("Final Loss:", loss)
print("Test Accuracy:", accuracy)


# Save detailed results

save_results(
    "initialization_results.csv",
    initialization_results,
    [
        "Initialization",
        "Initialization Seed",
        "Final Training Loss",
        "Test Accuracy"
    ]
)


# Save five-seed summary

save_results(
    "initialization_summary.csv",
    [[
        mean_loss,
        std_loss,
        mean_accuracy,
        std_accuracy
    ]],
    [
        "Mean Training Loss",
        "Std Training Loss",
        "Mean Test Accuracy",
        "Std Test Accuracy"
    ]
)


# Compare hidden neurons after zero initialization

print("\nComparing hidden neurons after zero initialization:")

for i in range(zero_model.hidden_size):

    for j in range(i + 1, zero_model.hidden_size):

        difference_W1 = np.max(
            np.abs(zero_model.W1[:, i] - zero_model.W1[:, j])
        )

        difference_b1 = abs(
            zero_model.b1[0, i] - zero_model.b1[0, j]
        )

        difference_W2 = np.max(
            np.abs(zero_model.W2[i, :] - zero_model.W2[j, :])
        )

        print(
            f"Neuron {i+1} vs Neuron {j+1}: "
            f"W1={difference_W1:.12e}, "
            f"b1={difference_b1:.12e}, "
            f"W2={difference_W2:.12e}"
        )
    
# ==========================================
# EXPERIMENT 2: LEARNING RATE
# ==========================================

print("\n" + "=" * 55)
print("EXPERIMENT 2: LEARNING RATE")
print("=" * 55)

learning_rates = [0.001, 0.01, 0.1, 1.0]

learning_rate_results = []
learning_rate_histories = {}

for lr in learning_rates:

    print("\nRunning learning rate:", lr)

    model, loss, accuracy = train_and_evaluate(
        hidden_size=4,
        learning_rate=lr,
        initialization="default",
        seed=SEED
    )

    learning_rate_histories[lr] = model.loss_history

    learning_rate_results.append([
        lr,
        loss,
        accuracy
    ])

    print("Final Training Loss:", loss)
    print("Test Accuracy:", accuracy)


# Plot learning curves
plt.figure(figsize=(10, 6))

for lr, history in learning_rate_histories.items():
    plt.plot(history, label=f"Learning rate = {lr}")

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "learning_rate_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

save_results(
    "learning_rate_results.csv",
    learning_rate_results,
    ["Learning Rate ", "Final Training Loss", "Test Accuracy"]
)


# ==========================================
# EXPERIMENT 3: NUMBER OF HIDDEN NEURONS
# ==========================================

print("\n" + "=" * 55)
print("EXPERIMENT 3: HIDDEN NEURONS")
print("=" * 55)

hidden_sizes = [1, 2, 4, 8, 16]

hidden_results = []


# Function to draw decision boundary
def plot_decision_boundary(model, hidden_size):

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    predictions = model.predict(grid)
    predictions = np.asarray(predictions).reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.3,
        levels=[-0.5, 0.5, 1.5]
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y.reshape(-1),
        edgecolors="black",
        s=25
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Decision Boundary - {hidden_size} Hidden Neurons")
    plt.grid(True)

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"decision_boundary_{hidden_size}_neurons.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


for hidden_size in hidden_sizes:

    print("\nRunning hidden neurons:", hidden_size)

    model, loss, accuracy = train_and_evaluate(
        hidden_size=hidden_size,
        learning_rate=0.1,
        initialization="default",
        seed=SEED
    )

    hidden_results.append([
        hidden_size,
        loss,
        accuracy
    ])

    print("Final Training Loss:", loss)
    print("Test Accuracy:", accuracy)

    plot_decision_boundary(model, hidden_size)


save_results(
    "hidden_neurons_results.csv",
    hidden_results,
    ["Hidden Neurons", "Final Training Loss", "Test Accuracy"]
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("STAGE 4 COMPLETED")
print("=" * 55)

print("\nInitialization Results:")
for row in initialization_results:
    print(row)

print("\nLearning Rate Results:")
for row in learning_rate_results:
    print(row)

print("\nHidden Neurons Results:")
for row in hidden_results:
    print(row)

print("\nAll outputs saved in:", OUTPUT_DIR)
