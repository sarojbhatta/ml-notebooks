'''
This dataset contains measurements on chemical properties, such as acidity, sugar, and alcohol content of white wines from northern Portugal.

Write a for loop to create 2000 bootstrap samples of the feature quality. Set random_state to 29.
Calculate the bootstrap sample proportions and save to the list bootWineList.
The code provided imports the dataset and packages and prints the distribution of values for both the original feature and the bootstrap samples.
'''

# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.utils import resample

# Load the wine_white dataset
wine = pd.read_csv('wine_white.csv')

bootWineList = []

# Write a for loop to create 2000 bootstrap samples of the feature quality
for i in range(2000):
    sample = resample(wine['quality'], random_state=29, replace=True) # Your code goes here
    

    # Calculate the bootstrap sample proportions and save to the list bootWineList
    # Your code goes here
    #oob = wine['quality'][~wine['quality'].index.isin(sample.index)]
    prop = sample.mean()
    
    bootWineList.append(prop) # Your code goes here

print(pd.DataFrame(bootWineList).describe())