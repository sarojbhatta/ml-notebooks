'''
This dataset contains biomedical voice measurements from 28 people with Parkinson's disease and eight without. 
The dataset consists of 24 input features of different types of voice measurements and one output feature indicating the existence of Parkinson's.

Initialize a t-distributed Stochastic Neighbor Embedding model in scikit-learn with five dimensions in the lower dimensional space, perplexity=23, random_state=171, and method='exact'.
Build a pipeline to scale the data and fit the t-SNE model to the scaled training data.
'''

# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the Parkinson's dataset
parkinsons = pd.read_csv('Parkinsons.csv')

X = parkinsons.drop(['name', 'status'], axis=1)
y = parkinsons[['status']]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize a t-SNE model with five dimensions in the lower dimensional space, 
# perplexity=23, random_state=171, and method='exact'
tsneParkinsons = TSNE(n_components=5, perplexity=23, random_state=171, method='exact') # Your code goes here 

# Build a pipeline to scale the data and fit the t-SNE model to the scaled training data
scaler = MinMaxScaler()
pipeline_tsne = Pipeline([('scaler', scaler), ('tsne', tsneParkinsons)]) # Your code goes here 

# Apply the pipeline
tsneParkinsonsTransform = pipeline_tsne.fit_transform(X_train) # Your code goes here

# Display the t-SNE KL Divergence and data points in lower-dimensional space
print("t-SNE KL Divergence:", tsneParkinsons.kl_divergence_)
print("Transformed Data:")
print(pd.DataFrame(tsneParkinsonsTransform))