# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Load the heart dataset
heart = pd.read_csv('heart.csv')

# Create a dataframe X containing chol and thalach
X = heart[['chol', 'thalach']]

# Output feature: target
y = heart[['target']]

# Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
heartModel = LinearSVC(C=1, penalty='l2', max_iter = 3000, random_state=8)

# Fit the model
heartModel.fit(X_train, np.ravel(y_train))

# Calculate confidence scores for testing set

print(heartModel.score(X_test, np.ravel(y_test))) # Your code goes here

# Calculate and print proportion of instances correctly classified for testing set
# Your code goes here
print(heartModel.decision_function(X_test))