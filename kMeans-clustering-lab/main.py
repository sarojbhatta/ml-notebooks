'''
The healthy_lifestyle dataset contains information on lifestyle measures such as amount of sunshine, pollution, 
and happiness levels for 44 major cities around the world. Apply k-means clustering to the cities' number of hours of sunshine and happiness levels.

Import the needed packages for clustering.
Initialize and fit a k-means clustering model using sklearn's Kmeans() function. 
Use the user-defined number of clusters, init='random', n_init=10, random_state=123, and algorithm='elkan'.
Find the cluster centroids and inertia.
Ex: If the input is:

4
the output should be:

Centroids: [[ 0.8294  0.2562]
 [ 1.3106 -1.887 ]
 [-0.9471  0.8281]
 [-0.6372 -0.7943]]
Inertia: 16.4991
'''


# Import needed packages
# Your code here
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

healthy = pd.read_csv('healthy_lifestyle.csv')

# Input the number of clusters
number = int(input())

# Define input features
X = healthy[['sunshine_hours', 'happiness_levels']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['sunshine_hours', 'happiness_levels'])
X = X.dropna()

# Initialize a k-means clustering algorithm with a user-defined number of clusters, init='random', n_init=10, 
# random_state=123, and algorithm='elkan'
# Your code here
kMeans = KMeans(n_clusters=number, init='random', n_init=10, random_state=123, algorithm='elkan')

# Fit the algorithm to the input features
# Your code here
kMeans.fit(X)

# Find and print the cluster centroids
centroid = kMeans.cluster_centers_ # Your code here
print("Centroids:", np.round(centroid,4))

# Find and print the cluster inertia
inertia = kMeans.inertia_ # Your code here
print("Inertia:", np.round(inertia,4))