import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import random

style.use('ggplot')


class K_Means:
        def __init__(self, k=3, max_iterations=1, centroids={}):
                self.k = k
                self.centroids = centroids
                self.max_iterations = max_iterations

        def fit(self, data):
                # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
                if self.centroids == {}:
                        for i in range(self.k):
                                self.centroids[i] = data[random.randint(0, len(data))]

                # begin iterations
                toReturn = {}
                for i in range(self.max_iterations):
                        self.classes = {}
                        for i in range(self.k):
                                self.classes[i] = []

                        # find the distance between the point and cluster; choose the nearest centroid
                        for features in data:
                                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                                classification = distances.index(min(distances))
                                self.classes[classification].append(features)

                        # previous = dict(self.centroids)

                        # average the cluster datapoints to re-calculate the centroids
                        for classification in self.classes:
                                self.centroids[classification] = np.average(self.classes[classification], axis=0)

                        for centroid in self.centroids:

                                # original_centroid = previous[centroid]
                                print(centroid+1, self.centroids[centroid])
                                toReturn[centroid] = self.centroids[centroid]
                                # curr = self.centroids[centroid]
                return toReturn

        def pred(self, data):
                distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                return classification


def main():
        df = pd.read_table(r"kmdata1.txt", delim_whitespace=True, names=['one', 'two'])

        X = df.values   # returns a numpy array
        km = K_Means(3, 10, {0: np.array([3.0, 3.0]), 1: np.array([6.0, 2.0]), 2: np.array([8.0,  5.0])})
        val = km.fit(X)
        for centroid in val:
                km.pred(val[centroid])

        # Plotting starts here
        colors = 10*["r", "g", "c", "b", "k"]

        for centroid in km.centroids:
                plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")

        for classification in km.classes:
                color = colors[classification]
                for features in km.classes[classification]:
                        plt.scatter(features[0], features[1], color=color, s=30)
        plt.show()


if __name__ == "__main__":
        main()
