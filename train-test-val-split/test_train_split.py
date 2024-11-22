'''
Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

Split the Dry_Bean_Data dataset into stratified 65% training/validation and 35% testing sets. Set the random_state parameter to 196. Set the stratify parameter to y.
Split the training/validation set into training and validation set so that the final split is 50% training, 15% validation, and 35% testing. Set the random_state parameter to 196.
'''

# Import packages and functions
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dry beans dataset
beans = pd.read_csv('Dry_Bean_Data.csv')

X = beans[['Extent', 'EquivDiameter']]
y = beans[['Class']]

# Set aside 35% of instances for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=196, stratify=y) # Your code goes here

# Split training again into 50% training and 15% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15/(1-0.35), random_state=196) # Your code goes here

# Print split sizes and test dataset
print('original dataset:', len(beans), 
    '\ntrain_data:', len(X_train), 
    '\nvalidation_data:', len(X_val), 
    '\ntest_data:', len(X_test),
    '\n', X_test
)