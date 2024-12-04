'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. 
Using a computer vision process, 16 features of each bean were extracted.

Construct a backward sequential feature selector with n_features_to_select='auto' and tol=None using the initialized estimator.
Construct a pipeline that scales the data and performs backward SFS with a linear discriminant analysis model.
Fit the model at the end of the pipeline using the training set.
'''


# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline

# Load the dry beans dataset
df = pd.read_csv('Dry_Bean_Dataset.csv')

X = df.drop(['Class'], axis=1)
y = df[['Class']]
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Construct a scaler
scalerFeat = StandardScaler()

# Construct an estimator
ldaEstimator = LinearDiscriminantAnalysis()

# Initialize a linear discriminant analysis model
ldaModel = LinearDiscriminantAnalysis()

# Construct a backward sequential feature selector using the initialized estimator
sfsBean = SequentialFeatureSelector(estimator=ldaEstimator, direction='backward', n_features_to_select='auto', tol=None) # Your code goes here 

# Construct a pipeline that scales the data and performs backward SFS and linear discriminant analysis
pipeSFS = Pipeline([('scaler', scalerFeat), ('sfs', sfsBean), ('model', ldaModel)])# Your code goes here 

# Fit the model at the end of the pipeline using the training set
pipeSFS.fit(X_train, np.ravel(y_train)) # Your code goes here

# Print the selected features 
print(X.columns[pipeSFS.named_steps['sfs'].support_])

# Print classification score
print(pipeSFS.score(X_test, y_test))