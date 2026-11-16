
import numpy as np

from mlp import CIPerceptron_main

np.set_printoptions(precision=8, suppress=True)


# =====================================
# PART 1: Manual Backpropagation
# =====================================

print("=" * 50)
print("MANUAL BACKPROPAGATION")
print("=" * 50)

model = CIPerceptron_main(
    input_size=2,
    hidden_size=2,
    output_size=1,
    learning_rate=0.1,
    seed=5
)

# Fixed weights
model.W1 = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

model.b1 = np.array([[0.1, 0.1]])

model.W2 = np.array([
    [0.5],
    [0.6]
])

model.b2 = np.array([[0.1]])

X = np.array([[0.5, 0.8]])
y = np.array([[1.0]])

# Forward propagation
output = model.feedforward(X)

print("\nInput:")
print(X)

print("\nHidden layer output:")
print(model.A1)

print("\nFinal output:")
print(output)

# Backpropagation
gradients = model.backpropagation(X, y)

print("\nCalculated gradients:")

for name, value in gradients.items():
    print(name)
    print(value)

# Save old weights
old_W1 = model.W1.copy()
old_W2 = model.W2.copy()

# One update step
model.update_weights(gradients)

print("\nUpdated W1:")
print(model.W1)

print("\nUpdated W2:")
print(model.W2)


# =====================================
# PART 2: Gradient Checking
# =====================================

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
