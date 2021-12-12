# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
                     
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

def read(filename):
    path = "."
    filename_read = os.path.join(path, filename)
    print(filename_read)
    # reads NA values as ?
    df = pd.read_csv(filename_read, na_values=['NA', '?']) 
    #selects only numerical coloumns drops symbol column drops date and symbol
    df = df.select_dtypes(include=['int', 'float'])
    return df

def heatmap(df):
    fig = plt.figure(figsize=(25, 20)) #scales the size of the png
    sns.heatmap(df.corr(), annot=True)
    plt.savefig("Heatmap.png")

def shuffling(df):
    np.random.seed(42) # Uncomment this line to get the same shuffle each time 
    df = df.reindex(np.random.permutation(df.index))
    df.reset_index(inplace=True, drop=True)
    return df

    #shuffling seems either not do much or greatly improve the result
    #could indicate the use of kfold split to get the best model

    #null_columns=df.columns[df.isnull().any()] #Contains 0 null columns

def pca(df):
    #if possible to auto generate X,y with all features instead of manually picking featuers for X
    X = df[['open','high','low','close','volume','divCash','splitFactor']].values.astype(np.float32)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
        
    #pca = PCA(n_components=None) #shows that only need 1 component to capture 100% of the data
    pca = PCA(n_components=None)
    pca.fit(X_scaled)
    
    # Get the eigenvalues
    print("Eigenvalues:")
    print(pca.explained_variance_)
    print()
    
    # Get explained variances
    print("Variances (Percentage):")
    print(pca.explained_variance_ratio_ * 100)
    print()

    # Make the scree plot
    plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.xlabel("Number of components (Dimensions)")
    plt.ylabel("Explained variance (%)")

def trained_pca(df):
    X = df[['open','high','low','close','volume','divCash','splitFactor']].values.astype(np.float32)
    y = df[['close']].values.astype(np.float32)
    
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)        
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    #standardises the data - will not do much as data is already similar in size
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    #fitting and testing the model
    pred = model.predict(X_test) #usually only out by 0.0005
    
    #shows no correlation between components 
    # X_pca = pd.DataFrame(X_pca)
    # fig = plt.figure(figsize=(10, 8))
    # sns.heatmap(X_pca.corr(), annot=True)
    
    return model, X_train, X_test, y_test, pred

def train(df):
    
    X = df[['open','high','low','close','volume','divCash','splitFactor']].values.astype(np.float32)
    y = df[['close']].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #standardises the data - will not do much as data is already similar in size
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    #fitting and testing the model
    pred = model.predict(X_test) #usually only out by 0.0005
    
    #shows corrolation between the first 4 components 
    # X = pd.DataFrame(X)
    # fig = plt.figure(figsize=(10, 8))
    # sns.heatmap(X.corr(), annot=True)
    
    return model, X_train, X_test, y_test, pred

def output(y_test,pred):
    
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))

    #code for plotting the predicted stock data against actual stock data
    plt.figure(figsize=(15, 5))
    
    plt.plot(1,2,1)
    plt.plot(np.array(y_test[0:5]))
    plt.plot(pred[0:5])
    
    plt.title('close values')
    plt.xlabel('close')
    
    print("Mean Squared error: {}".format(score)) 

def correlationtest():
    dataframe = read('GOOG.csv') #returns dataframe 
    #heatmap(dataframe) #adjvolume and volume are not corrolated to other covariates
    pca(dataframe) #only need the first component to capture almost 100% of the data

def runModel():
    dataframe = read('GOOG.csv') #returns dataframe 
    dataframe = shuffling(dataframe) #returns dataframe
    trained_model = train(dataframe) #return model, X_train, X_test, y_test, pred
    output(trained_model[3],trained_model[4])

def run_PCA_Model():
    dataframe = read('GOOG.csv') #returns dataframe 
    dataframe = shuffling(dataframe) #returns dataframe
    trained_model = trained_pca(dataframe) #return model, X_train, X_test, y_test, pred
    output(trained_model[3],trained_model[4])

#correlationtest()
runModel()
#run_PCA_Model() #has a higher R squared value but no corrleation, suggesting hidden variables???