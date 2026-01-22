
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


customer_data = pd.read_csv('/Users/vanshsaxena/Documents/Machine Learning Models/Customer Segmentation/data/Mall_Customers.csv')

#customer_data.head()
#customer_data.shape
#customer_data.info()
#customer_data.isnull().sum()

"""#### chosing the annual income column & Spending Score column"""
X = customer_data.iloc[:,[3,4]].values

"""### Chosing the number of cluster
### WCSS - Within Clusters Sum of Squares
"""
#finding wcss values for different number of clusters

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)

#Plot an elbow graph

sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show

"""### Optimum Number of clusters  5"""

# Training the K-Means Clustering Model

kmeans = KMeans(n_clusters=5,init='k-means++',random_state=0)

#return a label for each data point based on their cluster
Y = kmeans.fit_predict

Y = kmeans.fit_predict(X)
print(Y)

"""## VISAULIZATION ALL THE CLUSTERS"""

# plotting all the clusters and their Centroids

plt.figure(figsize=(8, 8))

plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=100,
    c='cyan',
    label='Centroids'
)

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

