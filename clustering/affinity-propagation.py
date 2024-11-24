'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. 
Using a computer vision process, 16 features of each bean were extracted.

Initialize an affinity propagation clustering model using scikit-learn with damping=0.5.
Fit the model to the standardized input features in X.
Print the indices of the cluster centers.
'''


# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation

# Load the dry beans dataset
bean = pd.read_csv('Dry_Bean_Dataset.csv')

# Subset input features Eccentricity and Solidity
X = bean[['Eccentricity', 'Solidity']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

# Initialize an affinity propagation clustering model using scikit-learn with damping=0.5
affinityModel = AffinityPropagation(damping=0.5) # Your code goes here 

# Fit the model to the input features
affinityModel.fit(X) # Your code goes here 

# Print the indices of the cluster centers.
indices = affinityModel.cluster_centers_indices_ # Your code goes here
print('Center indices:', indices)

# Print the cluster labels
print('Cluster labels:', affinityModel.labels_)