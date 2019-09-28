import pandas as pd
beer = pd.read_csv('data.txt', sep=' ')
print(beer)

X = beer[["calories", "sodium", "alcohol", "cost"]]

#K-means clustering

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=2).fit(X)
print(km.labels_)

beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_
beer.sort_values('cluster')

from pandas.tools.plotting import scatter_matrix

cluster_centers = km.cluster_centers_
cluster_centers = km2.cluster_centers_
beer.groupby("cluster").mean()
beer.groupby("cluster2").mean()
centers = beer.groupby("cluster").mean().reset_index()

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])

plt.scatter(beer["calories"], beer["alcohol"], c=colors[beer["cluster"]])
plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')

plt.xlabel('Calories')
plt.ylabel("Alcohol")
plt.show()

scatter_matrix(beer[["calories", "sodium", "alcohol", "cost"]], s=100, alpha=1, c=colors[beer["cluster"]], figsize=(10, 10))
plt.suptitle("With 3 centroids initialized")
plt.show()

scatter_matrix(beer[["calories", "sodium", "alcohol", "cost"]], s=100, alpha=1, c=colors[beer["cluster2"]], figsize=(10, 10))
plt.suptitle("with 2 centrids initialized")
plt.show()

#Scaled data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

km = KMeans(n_clusters=3).fit(X_scaled)
beer["scaled_cluster"] = km.labels_
beer.sort_values("scaled_cluster")
beer.groupby("scaled_cluster").mean()
pd.scatter_matrix(X, c=colors[beer.scaled_cluster], alpha=1, figsize=(10, 10), s=100)

#聚类评估：轮廓系数
from sklearn import metrics
score_scaled = metrics.silhouette_score(X, beer.scaled_cluster)
score = metrics.silhouette_score(X, beer.cluster)
print(score_scaled, score)

scores = []
for k in range(2, 20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)

print(scores)

plt.plot(list(range(2, 20)), scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=10, min_samples=2).fit(X)
lbels = db.labels_
beer['cluster_db'] = labels
beer.sort_values('cluster_db')
beer.groupby('cluster_db').mean()
pd.scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10, 10), s=100)
plt.show()