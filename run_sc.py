import pandas
import matplotlib.pyplot as pyplot

from sklearn.cluster import SpectralClustering


dataset = pandas.read_csv("dataset_moon.csv")

print(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatterplot_moon.png")
pyplot.close()

machine = SpectralClustering(n_clusters=2, n_components=4, gamma = 10, affinity="laplacian")

result = machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'], c=result)
pyplot.savefig("scatterplot_moon_color.png")
pyplot.close()







