'''
The diamonds.csv dataset contains the price, cut, color, and other characteristics of a sample of diamonds. The dataframe X contains all the features except cut, color, clarity, and price. The dataframe y contains the feature price. Using the Keras workflow, implement a neural net regressor in TensorFlow to predict price.

Set the backend using tensorflow.
Define the model structure using keras.Sequential:
The input layer has shape=(6, ).
Hidden layer 1 has 256 nodes and relu activation.
Hidden layer 2 had 128 nodes and linear activation.
Hidden layer 3 has 64 nodes and linear activation.
The output layer has 1 node (for regression) and linear activation.
Specify training choices using the compile method, with optimizer='Adam', loss='MeanSquaredError', and metrics='mse'.
Train the model with a batch size of 100, 5 epochs, validation_split=0.1, and verbose=0.
Ex: If a linear activation function is used in the first hidden layer instead of relu, the output is:

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 256)               1792      

 dense_1 (Dense)             (None, 128)               32896     

 dense_2 (Dense)             (None, 64)                8256      

 dense_3 (Dense)             (None, 1)                 65        

=================================================================
Total params: 43,009
Trainable params: 43,009
Non-trainable params: 0
_________________________________________________________________
None
Predictions: [[4891.566]
 [5190.253]
 [4927.398]]
Actual values:        price
11049    596
27577  18407
45498   1680
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Set the backend using tensorflow
# Your code here
import tensorflow
os.environ["KERAS_BACKEND"] = 'tensorflow'

# Suppress tensorflow INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# The backend must be set before importing keras, not after
import keras
keras.utils.set_random_seed(812)

df = pd.read_csv('diamonds.csv')
diamond_sample = df.sample(1000, random_state=12)

X = diamond_sample.drop(columns=['cut', 'color', 'clarity', 'price'])
y = diamond_sample[['price']]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

# Define the model structure using keras.Sequential. The input layer has shape=(6, ), hidden layer 1 has
# 256 nodes and relu activation, hidden layer 2 had 128 nodes and linear activation, hidden layer 3 has 
# 64 nodes and linear activation, and the output layer has 1 node (for regression) and linear activation

# Your code here
model = keras.Sequential([
    keras.layers.Input(shape=(6, )),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='linear'),
    keras.layers.Dense(64, activation='linear'),
    keras.layers.Dense(1, activation='linear')
])

# Include line_length=80 in model.summary() to format width of printed output
print(model.summary(line_length=80))


# Specify training choices using the compile method, with optimizer='Adam', loss='MeanSquaredError',
# metrics='mse'
# Your code here
model.compile(optimizer='adam', loss='MeanSquaredError', metrics=['mse'])

# Train the model with a batch size of 100, 5 epochs, validation_split=0.1, and verbose=0
# Your code here
model.fit(Xtrain, np.ravel(ytrain), batch_size=100, epochs=5, validation_split=0.1, verbose=0)

predictions = model.predict(Xtest[:3], verbose=0)
print('Predictions:', predictions.round(3))
print('Actual values:', ytest[:3])