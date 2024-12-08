'''
The healthy_lifestyle dataset contains information on lifestyle measures such as amount of sunshine, pollution, 
and happiness levels for 44 major cities around the world. 
Apply agglomerative clustering to the cities' number of hours of sunshine and happiness levels using both sklearn and SciPy.

Import the needed packages for agglomerative clustering from sklearn and SciPy.
Initialize and fit an agglomerative clustering model using sklearn's AgglomerativeClustering() function. 
Use the user-defined number of clusters and ward linkage.
Add cluster labels to the healthy dataframe.
Calculate the distances between all instances using SciPy's pdist() function.
Convert the distance matrix to a square matrix using SciPy's squareform() function.
Define a clustering model with ward linkage using SciPy's linkage() function.
Ex: If the input is:

4
the output should be:

   sunshine_hours  happiness_levels  labels
0          1858.0              7.44       3
1          2636.0              7.22       0
2          1884.0              7.29       3
3          1821.0              7.35       3
4          1630.0              7.64       3
First five rows of the linkage matrix from SciPy:
 [[39. 40.  0.  2.]
 [28. 43.  0.  3.]
 [ 7. 18.  0.  2.]
 [ 0.  3.  0.  2.]
 [ 8. 42.  0.  2.]]
 
 '''

import pandas as pd
import seaborn as sns
import numpy as np

# Import needed sklearn packages
# Your code here
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Import needed scipy packages
# Your code here
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, pdist

# Silence warning
import warnings
warnings.filterwarnings('ignore')


healthy = pd.read_csv('healthy_lifestyle.csv')
healthy = healthy.dropna(subset = ['sunshine_hours', 'happiness_levels'])

# Input the number of clusters
number = int(input())

# Define input features
X = healthy[['sunshine_hours', 'happiness_levels']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['sunshine_hours', 'happiness_levels'])

# Initialize and fit an agglomerative clustering model using ward linkage in scikit-learn, with a user-defined
# number of clusters
# Your code here
aggModel = AgglomerativeClustering(n_clusters=number, linkage='ward')

# Add cluster labels to the healthy dataframe
healthy['labels']= aggModel.fit_predict(X) # Your code here
print(healthy[['sunshine_hours', 'happiness_levels', 'labels']].head())

# Perform agglomerative clustering using SciPy
#aggModel.fit(X)

# Calculate the distances between all instances
# Your code here
dist = pdist(X)

# Convert the distance matrix to a square matrix
# Your code here
dist = squareform(dist)

# Define a clustering model with ward linkage
clustersHealthyScipy = linkage(dist, method='ward') # Your code here

print('First five rows of the linkage matrix from SciPy:\n', np.round(clustersHealthyScipy[:5, :], 0))