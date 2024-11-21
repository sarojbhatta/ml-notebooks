'''
The mpg_clean.csv dataset contains information on miles per gallon (mpg) and engine size for cars sold from 1970 through 1982. Dataframe X contains the input features mpg, cylinders, displacement, horsepower, weight, acceleration, and model_year. Dataframe y contains the output feature origin.

Initialize and fit a random forest classifier with a user-input number of decision trees, estimator, a user-input number of features considered at each split, max_features, and a random state of 123.
Calculate the prediction accuracy for the model.
Read the documentation for the permutation_importance function from scikit-learn's inspection module.
Calculate the permutation importance using the default parameters and a random state of 123.
Ex: When the input is

5
3
the output is:

0.9796
        feature  permutation importance
2  displacement                0.453571
0           mpg                0.160204
4        weight                0.133673
3    horsepower                0.107653
5  acceleration                0.057143
6    model_year                0.051531
1     cylinders                0.012245

'''

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

df = pd.read_csv('mpg_clean.csv')

# Create a dataframe X containing the input features
X = df.drop(columns=['name', 'origin'])
# Create a dataframe y containing the output feature origin
y = df[['origin']]

# Get user-input n_estimators and max_features
estimators = int(input())
max_features = int(input())

# Initialize and fit a random forest classifier with user-input number of decision trees, 
# user-input number of features considered at each split, and a random state of 123
rfModel =  RandomForestClassifier(n_estimators=estimators, max_features=max_features, random_state=123) # Your code here
rfModel.fit(X, np.ravel(y)) # Your code here

# Calculate prediction accuracy
score = rfModel.score(X, np.ravel(y)) # Your code here
print(round(score, 4))

# Calculate the permutation importance using the default parameters and a random state of 123
result = permutation_importance(rfModel, X, y, random_state=123) # Your code here

# Variable importance table
importance_table = pd.DataFrame(
    data={'feature': rfModel.feature_names_in_,'permutation importance': result.importances_mean}
).sort_values('permutation importance', ascending=False)

print(importance_table)