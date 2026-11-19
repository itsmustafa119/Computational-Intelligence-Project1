

import numpy as np

from mlp import CIPerceptron_main

np.set_printoptions(precision=8, suppress=True)


# ==========================================
# PART 1: Manual Backpropagation Verification
# ==========================================

print("=" * 60)
print("MANUAL BACKPROPAGATION VERIFICATION")
print("=" * 60)


# ------------------------------------------
# 1. Initialize network
# ------------------------------------------

learning_rate = 0.1

model = CIPerceptron_main(
    input_size=2,
    hidden_size=2,
    output_size=1,
    learning_rate=learning_rate,
    seed=5
)

# Fixed initial weights and biases

W1 = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

b1 = np.array([[0.1, 0.1]])

W2 = np.array([
    [0.5],
    [0.6]
])

b2 = np.array([[0.1]])

model.W1 = W1.copy()
model.b1 = b1.copy()
model.W2 = W2.copy()
model.b2 = b2.copy()

X = np.array([[0.5, 0.8]])
y = np.array([[1.0]])


# ------------------------------------------
# 2. Manual Feedforward
# ------------------------------------------

print("\n--- MANUAL FEEDFORWARD ---")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Hidden layer

Z1_manual = X @ W1 + b1
A1_manual = sigmoid(Z1_manual)

# Output layer

Z2_manual = A1_manual @ W2 + b2
A2_manual = sigmoid(Z2_manual)

print("Z1:")
print(Z1_manual)

print("A1:")
print(A1_manual)

print("Z2:")
print(Z2_manual)

print("Final output:")
print(A2_manual)

# Manual SSE

manual_loss = np.sum((A2_manual - y) ** 2)

print("SSE:")
print(manual_loss)


# ------------------------------------------
# 3. Manual Backpropagation
# ------------------------------------------

print("\n--- MANUAL BACKPROPAGATION ---")

# Output layer error signal

delta2_manual = (
    2 * (A2_manual - y)
    * A2_manual * (1 - A2_manual)
)

# Hidden layer error signal

delta1_manual = (
    delta2_manual @ W2.T
    * A1_manual * (1 - A1_manual)
)

# Weight gradients

dW2_manual = A1_manual.T @ delta2_manual
db2_manual = delta2_manual.copy()

dW1_manual = X.T @ delta1_manual
db1_manual = delta1_manual.copy()

manual_gradients = {
    "dW1": dW1_manual,
    "db1": db1_manual,
    "dW2": dW2_manual,
    "db2": db2_manual
}

print("Delta output:")
print(delta2_manual)

print("Delta hidden:")
print(delta1_manual)

print("\nManual gradients:")

for name, value in manual_gradients.items():
    print(name)
    print(value)


# ------------------------------------------
# 4. Get gradients from actual MLP code
# ------------------------------------------

print("\n--- MLP CODE RESULTS ---")

output_code = model.feedforward(X)

gradients_code = model.backpropagation(X, y)

print("Network output:")
print(output_code)

print("\nGradients from MLP:")

for name, value in gradients_code.items():
    print(name)
    print(value)


# ------------------------------------------
# 5. Compare manual and code results
# ------------------------------------------

print("\n--- MANUAL VS CODE COMPARISON ---")

comparisons = {
    "Hidden output A1": (A1_manual, model.A1),
    "Final output A2": (A2_manual, output_code),
    "Gradient dW1": (
        dW1_manual, gradients_code["dW1"]
    ),
    "Gradient db1": (
        db1_manual, gradients_code["db1"]
    ),
    "Gradient dW2": (
        dW2_manual, gradients_code["dW2"]
    ),
    "Gradient db2": (
        db2_manual, gradients_code["db2"]
    )
}

all_passed = True

for name, (manual, code) in comparisons.items():

    difference = np.max(np.abs(manual - code))

    passed = np.allclose(
        manual,
        code,
        atol=1e-10,
        rtol=1e-10
    )

    print(f"\n{name}")
    print("Manual:")
    print(manual)

    print("MLP code:")
    print(code)

    print("Maximum absolute difference:", difference)

    print("Result:", "PASSED" if passed else "FAILED")

    if not passed:
        all_passed = False


# ------------------------------------------
# 6. Manual weight update
# ------------------------------------------

print("\n--- MANUAL WEIGHT UPDATE ---")

W1_updated_manual = W1 - learning_rate * dW1_manual
b1_updated_manual = b1 - learning_rate * db1_manual

W2_updated_manual = W2 - learning_rate * dW2_manual
b2_updated_manual = b2 - learning_rate * db2_manual


# Save original parameters before code update

model.W1 = W1.copy()
model.b1 = b1.copy()
model.W2 = W2.copy()
model.b2 = b2.copy()

# Update using actual MLP function

model.update_weights(gradients_code)


updated_comparisons = {
    "Updated W1": (W1_updated_manual, model.W1),
    "Updated b1": (b1_updated_manual, model.b1),
    "Updated W2": (W2_updated_manual, model.W2),
    "Updated b2": (b2_updated_manual, model.b2)
}

print("\nManual vs MLP updated parameters:")

for name, (manual, code) in updated_comparisons.items():

    difference = np.max(np.abs(manual - code))

    passed = np.allclose(
        manual,
        code,
        atol=1e-10,
        rtol=1e-10
    )

    print(f"\n{name}")
    print("Manual:")
    print(manual)

    print("MLP code:")
    print(code)

    print("Maximum absolute difference:", difference)

    print("Result:", "PASSED" if passed else "FAILED")

    if not passed:
        all_passed = False


# ------------------------------------------
# 7. Final verification result
# ------------------------------------------

print("\n" + "=" * 60)

if all_passed:
    print("MANUAL BACKPROPAGATION VERIFICATION PASSED")
    print("Manual calculations match the MLP implementation.")
else:
    print("MANUAL BACKPROPAGATION VERIFICATION FAILED")
    print("There are differences between manual and code results.")

print("=" * 60)


# ==========================================
# PART 2: Gradient Checking
# Keep your existing Gradient Checking code
# below this line.
# ==========================================


print("\n" + "=" * 50)
print("GRADIENT CHECKING")
print("=" * 50)

model = CIPerceptron_main(
    input_size=2,
    hidden_size=2,
    output_size=1,
    learning_rate=0.1,
    seed=5
)

X = np.array([[0.5, 0.8]])
y = np.array([[1.0]])

# Analytical gradients
analytical = model.backpropagation(X, y)

epsilon = 1e-5

# Numerical gradient function
def numerical_gradient(parameter_name):

    parameter = getattr(model, parameter_name)

    numerical = np.zeros_like(parameter)

    for index in np.ndindex(parameter.shape):

        original = parameter[index]

        parameter[index] = original + epsilon
        loss_plus = model.calculate_loss(X, y)

        parameter[index] = original - epsilon
        loss_minus = model.calculate_loss(X, y)

        parameter[index] = original

        numerical[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

    return numerical


parameter_mapping = {
    "W1": "dW1",
    "b1": "db1",
    "W2": "dW2",
    "b2": "db2"
}

total_error = 0
total_count = 0

for parameter_name, gradient_name in parameter_mapping.items():

    numerical = numerical_gradient(parameter_name)

    analytical_gradient = analytical[gradient_name]

    difference = np.abs(
        numerical - analytical_gradient
    )

    print("\nParameter:", parameter_name)

    print("Analytical gradient:")
    print(analytical_gradient)

    print("Numerical gradient:")
    print(numerical)

    print("Maximum difference:")
    print(np.max(difference))

    total_error += np.sum(difference)
    total_count += difference.size


mean_error = total_error / total_count

print("\nFinal Mean Absolute Difference:")
print(mean_error)

print("\nNumerical step size:")
print(epsilon)

if mean_error < 1e-6:
    print("Gradient Checking PASSED")
else:
    print("Gradient Checking needs investigation")
