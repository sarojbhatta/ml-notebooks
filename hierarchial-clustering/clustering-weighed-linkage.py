'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. 
Using a computer vision process, 16 features of each bean were extracted. 
Create an agglomerative clustering model using scipy.

Calculate the distances between all instances in X.
Convert the distances into a square matrix.
Define a clustering model with weighted linkage.
'''


# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage, ClusterWarning
from scipy.spatial.distance import squareform, pdist
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

# Load the dry beans dataset
bean = pd.read_csv('Dry_Bean_Dataset.csv')

# Subset input features Solidity and Extent
X = bean[['Solidity', 'Extent']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

# Calculate distances between all instances
beanDist = pdist(X) # Your code goes here

# Convert distances into a square matrix
beanDist = squareform(beanDist) # Your code goes here

# Define a clustering model with weighted linkage
clustersBean = linkage(beanDist, method='weighted') # Your code goes here

# Print the hierarchical clustering
print(clustersBean[:5, :])