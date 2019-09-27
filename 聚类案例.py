import pandas as pd
import pandas as pd
#KMeans
beer = pd.read_csv('data.txt', sep=' ')

X = beer[["calories", "sodium", "alcohol", "cost"]]

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=2).fit(X)

print(km.labels_)

beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_
beer.sort_values('cluster')

from pandas.tools.plotting import scatter_matrix

cluster_centers = km.cluster_centers_
cluster_centers_2 = km2.cluster_centers_
print(beer.groupby("cluster").mean())
print(beer.groupby("cluster2").mean())

centers = beer.groupby("cluster").mean().reset_index()
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.show()

import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])

plt.scatter(beer["calories"], beer["alcohol"], c=colors[beer["cluster"]])
plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')
plt.xlabel("Calories")
plt.ylabel("Alcohol")

scatter_matrix(beer[["calories", "sodium", "alcohol", "cost"]], s=100, alpha=1, c=colors[beer["cluster"]], figsize=(10,5))
plt.suptitle("With 3 centroids initialized")

scatter_matrix(beer[["calories", "sodium", "alcohol", "cost"]], s=100, alpha=1, c=colors[beer["cluster2"]], figsize=(10,5))
plt.suptitle("With 2 centroids initialized")
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

km = KMeans(n_clusters=3).fit(X_scaled)
beer["scaled_cluster"] = km.labels_
beer.sort_values("scaled_cluster")

beer.groupby("scaled_cluster").mean()
pd.scatter_matrix(X, c=colors[beer.scaled_cluster], alpha=1, figsize=(10, 10), s=100)
plt.show()

#轮廓系数
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
plt.show()

#DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=10, min_samples=2).fit(X)
labels = db.labels_
beer['cluster_db'] = labels
beer.sort_values('cluster_db')
beer.groupby('cluster_db').mean()
pd.scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10, 10), s=100)
plt.show()