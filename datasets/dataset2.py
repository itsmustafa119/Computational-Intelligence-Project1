import matplotlib as plt
from sklearn.datasets import make_circles

X, y = make_circles(
    n_samples=300,
    noise=0.05,
    factor=0.5,
    random_state=42
)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset 2 - Circles")
plt.show()