'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. 
Using a computer vision process, 16 features of each bean were extracted.

Initialize a mean-shift clustering model using scikit-learn with a bandwidth of 2.
Fit the model to the standardized input features in X.
Print the cluster centers.
'''

import warnings
warnings.filterwarnings("ignore")

# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

# Load the dry beans dataset
bean = pd.read_csv('Dry_Bean_Dataset.csv')

# Subset input features Extent and Solidity
X = bean[['Extent', 'Solidity']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

# Initialize a mean-shift clustering model in scikit-learn with a bandwidth of 2
meanshiftBean = MeanShift(bandwidth=2) # Your code goes here 

# Fit the model to the input features
meanshiftBean.fit(X) # Your code goes here 

# Print the cluster centers
centers = meanshiftBean.cluster_centers_ # Your code goes here
print('Centers:', centers)

# Print the cluster labels
print('Cluster labels:', meanshiftBean.labels_)