import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

sleep = pd.read_csv('msleep_clean.csv')

# Create a dataframe X containing the features awake, brainwt, and bodywt, in that order
# Your code here
X = sleep[['awake', 'brainwt', 'bodywt']]

# Output feature: sleep_rem
# Your code here
y = sleep['sleep_rem']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize the model with max_depth=3, ccp_alpha=0.02, and random_state=123
DTRModel = DecisionTreeRegressor(max_depth=3, ccp_alpha=0.02, random_state=123) # Your code here

# Fit the model
# Your code here
DTRModel.fit(X_train, y_train)

# Print the R-squared value for the testing set
# Your code here
print(r2_score(y_test, DTRModel.predict(X_test)))

# Print text summary of tree
DTR_tree = export_text(DTRModel)
print(DTR_tree)