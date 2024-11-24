'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. 
Using a computer vision process, 16 features of each bean were extracted.

Initialize a DBSCAN clustering model using scikit-learn with 
 = 1.5 and 
 = 14.
Fit the model to the standardized input features in X.
Print the cluster labels.
'''

# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load the dry beans dataset
bean = pd.read_csv('Dry_Bean_Dataset.csv')

# Subset input features Extent and Compactness
X = bean[['Extent', 'Compactness']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

# Initialize a DBSCAN clustering model in scikit-learn with epsilon = 1.5, m = 14
dbscanBean = DBSCAN(eps=1.5, min_samples=14) # Your code goes here

# Fit the model to the input features
dbscanBean.fit(X) # Your code goes here 

# Print the cluster labels 
labels = dbscanBean.labels_ # Your code goes here
print('Labels:', labels)

# Print the number of core points 
print('Number of core points:', len(dbscanBean.core_sample_indices_))

# Print parameter values
print(dbscanBean.get_params())