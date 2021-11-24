# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:19:26 2021

@author: dhruv
"""

import os
import pandas as pd
import codecs
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


# Convert a Pandas dataframe to the X,y inputs that Keras needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)


path = "."
filename_read = os.path.join(path, "GOOG.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])  # reads NA values as ?
df = df.select_dtypes(include=['int', 'float'])  # selects only numerical coloumns drops symbol column

# np.random.seed(42) # Uncomment this line to get the same shuffle each time
# shuffling has not effect on the data
df = df.reindex(np.random.permutation(df.index))
df.reset_index(inplace=True, drop=True)

# null_columns=df.columns[df.isnull().any()] #Contains 0 null columns

X, y = to_xy(df, "close")  # Predicting the close value
y = tf.keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardises the data - improves like crazy
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# building the model with 1 hidden node -> 1 sigmoid
model = Sequential()
model.add(Dense(1, input_dim=X.shape[1], activation='sigmoid'))  # Hidden 1 using sigmoid
# model.add(Dropout(0.1)) #makes the graph look all weird
model.add(Dense(y.shape[1]))  # Output
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='loss', min_delta=0.1, patience=5)  # Does not have an effect
model.summary()
model.fit(X, y, verbose=2, epochs=50)  # trains the just as well on a low number of epochs 5 is the same as 50

# plot the loss on the training data, and also the validation data
plt.figure(figsize=(10, 10))  # both loss and val_loss follow the same pattern
training_trace = model.fit(X_train, y_train, callbacks=monitor, validation_split=0.2, verbose=2, epochs=50)
plt.plot(training_trace.history['loss'])
plt.plot(training_trace.history['val_loss'])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

# fitting and testing the model
pred = model.predict(X_test)
score = np.sqrt(metrics.mean_squared_error(pred, y_test))

print("Mean Squared error: {}".format(score))  # 0.0199 aka really good
