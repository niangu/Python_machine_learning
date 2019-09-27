import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Fremont.csv', index_col='Date', parse_dates=True)
print(data.head())

#Visualizing the Dataset
data.plot()
data.resample('w').sum().plot()
data.resample('D').sum().rolling(365).sum().plot()
plt.show()

import matplotlib.pyplot as plt
data.groupby(data.index.time).mean().plot()
plt.xticks(rotation=45)
plt.show()
data.columns = ['West', 'East']
data['Total'] = data['West'] + data['East']
pivoted = data.pivot_table('Total', index=data.index.time, columns=data.index.date)
print(pivoted.iloc[:5, :5])
pivoted.plot(legend=False, alpha=0.01)
plt.xticks(rotation=45)
plt.show()
print(pivoted.shape)
X = pivoted.fillna(0).T.values
print(X.shape)
from sklearn.decomposition import PCA
X2 = PCA(2).fit_transform(X)
print(X2.shape)
plt.scatter(X2[:, 0], X2[:, 1])
plt.show()

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(2)
gmm.fit(X)
labels = gmm.predict_proba(X)
print(labels)

labels = gmm.predict(X)
print(labels)

plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='rainbow')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

pivoted.T[labels == 0].T.plot(legend=False, alpha=0.1, ax=ax[0])
pivoted.T[labels == 1].T.plot(legend=False, alpha=0.1, ax=ax[1])

ax[0].set_title('Purple Cluster')
ax[1].set_title('Red Cluster')
plt.show()

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=800, centers=4, random_state=11)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.show()

import numpy as np
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=1)
kmeans.fit(X_stretched)
y_kmeans = kmeans.predict(X_stretched)
plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.show()

gmm = GaussianMixture(n_components=4)
gmm.fit(X_stretched)
y_gmm = gmm.predict(X_stretched)
plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=y_gmm, s=50, cmap='viridis')
plt.show()