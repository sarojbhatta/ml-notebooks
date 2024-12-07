'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. 
Using a computer vision process, 16 features of each bean were extracted.

Create a principal component analysis model using the PCA() function with 4 components and random_state = 12.
Create a data pipeline with a scaler, pca model, and a support vector classifier, and fit the model at the end of the pipeline using the training set.
Get the principal components.

'''

# Import packages and functions
from sklearn.svm import SVC
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Load the dry beans dataset
bean = pd.read_csv('Dry_Bean_Dataset.csv')

X = bean.drop(['Class'], axis=1)
y = bean[['Class']]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create a support vector classifier
clf = SVC(gamma='scale', class_weight='balanced', C=50, random_state=123)

# Create a scaler to standardize input features
scaler = StandardScaler()

# Create a principal component analysis model using the PCA() function with 4 
# components and random_state = 12
pcaBean = PCA(n_components=4, random_state=12) # Your code goes here 

# Create a data pipeline with a scaler, pca model, and a support vector classifier
beanPipeline = Pipeline([('scaler', scaler), ('pcamodel', pcaBean), ('svc', clf)]) # Your code goes here 

# Fit the model at the end of the pipeline using the training set
beanPipeline.fit(X_train, np.ravel(y_train)) # Your code goes here

# Get the principal components
componentsBean = pcaBean.components_ # Your code goes here

# Print the principal components and accuracy
print(componentsBean)
print('Accuracy is:', beanPipeline.score(X_test, y_test))