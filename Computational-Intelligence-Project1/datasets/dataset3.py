import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class Dataset3:
    def create(self):    
        self.X, self.y = make_blobs(
                n_samples=300,
                n_features=2,
                centers=3,
                cluster_std=1.0,
                random_state=119
            )
        return self.X, self.y
          
    def draw(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap="coolwarm")

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Dataset 1 - Linear Classification")
    def show():
        plt.show()

    def save():
        plt.savefig("outputs/dataset1_scatter_plot", dpi = 300, bbox_inches = "tight")