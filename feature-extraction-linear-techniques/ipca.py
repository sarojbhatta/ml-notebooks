'''
Create an ICA model using the FastICA() function with three components, algorithm='deflation', and whiten='unit-variance'.
Fit and transform the mixed signals using the FastICA model.
Get the unmixing matrix.
'''


# Import packages and functions
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Randomly generate original sources
np.random.seed(54)

samples = 2000
time = np.linspace(0, 10, samples)

signal_1 = np.cos(time)
signal_2 = signal.square(3*np.pi*time) 
signal_3 = signal.sawtooth(2*np.pi*time) 

S = np.c_[signal_1, signal_2, signal_3]

# Specify mixing matrix and compute mixed signals
A = np.array([[2, 0.4, 1], [3, 0.1, 0.5], [0.4, 2, 3]])
X = np.dot(S, A.T)

# Create a FastICA model
threeSourceICA = FastICA(n_components=3, algorithm='deflation', whiten='unit-variance') # Your code goes here 

# Fit and transform the mixed signals using the FastICA model
estimatedSource = threeSourceICA.fit_transform(X) # Your code goes here 

# Print the first five elements of the estimated source signal
print(estimatedSource[0:5])

# Get the unmixing matrix
matrixUnmix = threeSourceICA.components_ # Your code goes here

# Print unmixing matrix
print(matrixUnmix)