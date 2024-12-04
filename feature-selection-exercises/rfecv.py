'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. 
Using a computer vision process, 16 features of each bean were extracted.

Construct a function to perform recursive feature elimination with cross-validation with cv=8 using the initialized estimator.
Construct a pipeline that scales the data and performs RFECV.
Fit the model at the end of the pipeline using the training set.
'''

# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

# Load the dry beans dataset
bean = pd.read_csv('Dry_Bean_Dataset.csv')

X = bean.drop(['Class'], axis=1)
y = bean[['Class']]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Construct a scaler
scalerFeat = StandardScaler()

# Construct an estimator
estimatorBean = LinearDiscriminantAnalysis()

# Perform recursive feature elimination with cross-validation with cv=8 using the given estimator
beanRFECV = RFECV(estimator=estimatorBean, cv=8, step=1) # Your code goes here 

# Construct a pipeline that scales the data and performs RFECV
pipeRFECV = Pipeline(steps=[('scaler', scalerFeat), ('rfecv', beanRFECV)]) # Your code goes here 

# Fit the model at the end of the pipeline using the training set
pipeRFECV.fit(X_train, np.ravel(y_train)) # Your code goes here

# Print classification score
print(pipeRFECV.score(X_test, y_test))