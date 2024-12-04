'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. 
Using a computer vision process, 16 features of each bean were extracted.

Perform feature selection by initializing and fitting a SelectPercentile() function, with score_func = f_classif and percentile = 20.
Get the features selected by the function.
'''

# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

# Load the dry beans dataset
bean = pd.read_csv('Dry_Bean_Dataset.csv')

X = bean.drop(['Class'], axis=1)
y = bean[['Class']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

# Initialize a SelectPercentile() function in scikit-learn with score_func = f_classif and percentile = 20
beanPerc = SelectPercentile(score_func=f_classif, percentile=20) # Your code goes here 

# Fit and transform the function
percBeanFeatures = beanPerc.fit_transform(X_train, np.ravel(y_train)) # Your code goes here 

# Get the features selected by the function
featureFilter = beanPerc.get_support() # Your code goes here

# Get input feature names
features = np.array(X_train.columns)

# Print feature names selected by the SelectPercentile() function 
print(features[featureFilter])