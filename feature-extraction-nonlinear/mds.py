'''
This dataset contains biomedical voice measurements from 28 people with Parkinson's disease and eight without. 
The dataset consists of 24 input features of different types of voice measurements and one output feature indicating the existence of Parkinson's.

Initialize a multidimensional scaling model in scikit-learn with five dimensions in the lower dimensional space and random_state=414.
Build a pipeline to scale the data and fit the MDS model to the scaled training data.
'''
# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.manifold import MDS

# Load the Parkinson's dataset
parkinsons = pd.read_csv('Parkinsons.csv')

X = parkinsons.drop(['name', 'status'], axis=1)
y = parkinsons[['status']]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize a multidimensional scaling model with five dimensions in the lower 
# dimensional space and random_state=414
parkinsonsMDS = MDS(n_components=5, random_state=414) # Your code goes here 

# Build a pipeline to scale the data and fit the MDS model to the scaled training data
scaler = MinMaxScaler()
pipeline_mds = Pipeline([('scaler', scaler), ('mds', parkinsonsMDS)]) # Your code goes here 

# Apply the pipeline
parkinsonsMDSTransform = pipeline_mds.fit_transform(X_train) # Your code goes here

# Display the raw stress
print(parkinsonsMDS.stress_)