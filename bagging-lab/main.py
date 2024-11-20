import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor

df = pd.read_csv('msleep_clean.csv')

# Create a dataframe X containing the features awake, brainwt, and bodywt, in that order
# Your code here
X = df[['awake', 'brainwt', 'bodywt']]

# Create a dataframe y containing sleep_rem
# Your code here
y = df['sleep_rem']

# Initialize and fit bagging regressor with 30 base estimators, a random state of 10, and oob_score=True
sleepModel = BaggingRegressor(n_estimators=30, random_state=10, oob_score=True) # Your code here
sleepModel.fit(X, np.ravel(y)) # Your code here

# Calculate out-of-bag accuracy
print(np.round(sleepModel.oob_score_, 4))

# Calculate predictions from out-of-bag estimate
print(np.round(sleepModel.oob_prediction_, 4))