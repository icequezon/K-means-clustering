
import numpy as np
import matplotlib.pyplot as plt, mpld3
from matplotlib import style
import pandas as pd 

style.use('ggplot')

class K_Means:
	def __init__(self, k =3, max_iterations = 100):
		self.k = k
		self.centroids = {0: np.array([3.0, 3.0]), 1: np.array([6.0, 2.0]), 2: np.array([8.0,  5.0])}
		self.max_iterations = max_iterations

	def fit(self, data):

		#begin iterations
		for i in range(self.max_iterations):
			self.classes = {}
			for i in range(self.k):
				self.classes[i] = []

			#find the distance between the point and cluster; choose the nearest centroid
			for features in data:
				distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classes[classification].append(features)

			previous = dict(self.centroids)

			#average the cluster datapoints to re-calculate the centroids
			for classification in self.classes:
				self.centroids[classification] = np.average(self.classes[classification], axis = 0)

			isOptimal = True

			for centroid in self.centroids:

				original_centroid = previous[centroid]
				curr = self.centroids[centroid]

	def pred(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

def main():
	
	df = pd.read_csv(r"kmdata1.txt", delim_whitespace=True)
	#df = df[["one", "two"]]
	dataset = df.astype(float).values.tolist()

	X = df.values #returns a numpy array
	
	km = K_Means(3)
	km.fit(X)

	# Plotting starts here
	colors = 10*["r", "g", "c", "b", "k"]

	for centroid in km.centroids:
		plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s = 130, marker = "x")

	for classification in km.classes:
		color = colors[classification]
		for features in km.classes[classification]:
			plt.scatter(features[0], features[1], color = color,s = 30)
	
	plt.show()

if __name__ == "__main__":
	main()
