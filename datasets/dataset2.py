import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


class Dataset2:
    def create(self):    
        self.X, self.y = make_circles(
                                            n_samples=300,
                                            noise=0.05,
                                            factor=0.5,
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