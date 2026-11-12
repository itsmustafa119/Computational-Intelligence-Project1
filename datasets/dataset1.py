import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


class Dataset1:
    def create(self):    
        self.X, self.y = make_classification(n_samples=300, 
                                    n_features=2, 
                                    n_informative=2, 
                                    n_redundant=0, 
                                    n_clusters_per_class=1,
                                    class_sep = 1, 
                                    random_state=119)
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


