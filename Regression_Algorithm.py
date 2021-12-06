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
import matplotlib.pyplot as plt


#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
                     
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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression


def read(filename):
    path = "."
    filename_read = os.path.join(path, filename)
    df = pd.read_csv(filename_read, na_values=['NA', '?']) # reads NA values as ?
    df = df.select_dtypes(include=['int', 'float']) #selects only numerical coloumns drops symbol column drops date and symbol
    return df


def shuffling(df):
    #np.random.seed(42) # Uncomment this line to get the same shuffle each time 
    #shuffling has not effect on the data
    df = df.reindex(np.random.permutation(df.index))
    df.reset_index(inplace=True, drop=True)
    return df

#null_columns=df.columns[df.isnull().any()] #Contains 0 null columns

def train(df):
    
    X = df[['open','high','low','close']].values.astype(np.float32)
    y = df[['close']].values.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #standardises the data - will not do much as data is already similar in size
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    #fitting and testing the model
    pred = model.predict(X_test) #usually only out by 0.0005
    
    return model, X_train, X_test, y_test, pred

def output(y_test,pred):
    
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))

    plt.figure(figsize=(15, 5))
    
    plt.plot(1,2,1)
    plt.plot(np.array(y_test[0:20]))
    plt.plot(pred[0:20])
    
    plt.title('close values')
    plt.xlabel('close')
    
    print("Mean Squared error: {}".format(score)) #0.0199 aka really good

dataframe = read('GOOG.csv') #returns dataframe 
dataframe = shuffling(dataframe) #returns dataframe
trained_model = train(dataframe) #return model, X_train, X_test, y_test, pred
output(trained_model[3],trained_model[4])