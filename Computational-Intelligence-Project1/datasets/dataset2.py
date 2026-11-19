"""Canonical generation and split settings for Dataset 2 (concentric circles)."""

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Keep these values shared by Stages 3 and 6 (and the Boosting experiment).
DATASET2_N_SAMPLES = 300
DATASET2_NOISE = 0.05
DATASET2_FACTOR = 0.5
DATASET2_SEED = 119

TEST_SIZE = 0.20
SPLIT_SEED = 7349


class Dataset2:
    def create(self):
        self.X, self.y = make_circles(
            n_samples=DATASET2_N_SAMPLES,
            noise=DATASET2_NOISE,
            factor=DATASET2_FACTOR,
            random_state=DATASET2_SEED,
        )
        return self.X, self.y

    def draw(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap="coolwarm")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Dataset 2 - Concentric Circles")

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def save():
        plt.savefig("outputs/dataset2_scatter_plot.png", dpi=300, bbox_inches="tight")
