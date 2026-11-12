import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=300,
    n_features=2,
    centers=3,
    cluster_std=1.0,
    random_state=119
)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset 3 - Blobs")
plt.show()

plt.savefig("outputs/dataset3_scatter_plot", dpi = 300, bbox_inches = "tight")