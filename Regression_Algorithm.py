# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


def read(filename):
    path = "."
    filename_read = os.path.join(path, filename)
    # reads NA values as ?
    df = pd.read_csv(filename_read, na_values=['NA', '?'])
    # selects only numerical columns drops symbol column drops date and symbol
    df = df.select_dtypes(include=['int', 'float'])
    return df


def heatmap(df):
    plt.figure(figsize=(25, 20))  # scales the size of the png
    sns.heatmap(df.corr(), annot=True)
    plt.savefig("Heatmap.png")


def shuffling(df):
    np.random.seed(42)  # Uncomment this line to get the same shuffle each time
    df = df.reindex(np.random.permutation(df.index))
    df.reset_index(inplace=True, drop=True)
    return df

    # shuffling seems either not do much or greatly improve the result
    # could indicate the use of kfold split to get the best model

    # null_columns=df.columns[df.isnull().any()] #Contains 0 null columns


def pca(df):
    # if possible to auto generate X,y with all features instead of manually picking featuers for X
    X = df[['open', 'high', 'low', 'close', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    # pca = PCA(n_components=None) #shows that only need 1 component to capture 100% of the data
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
    X = df[['open', 'high', 'low', 'close', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)
    y = df[['close']].values.astype(np.float32)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # standardises the data - will not do much as data is already similar in size

    model = LinearRegression()
    model.fit(X_train, y_train)

    # fitting and testing the model
    pred = model.predict(X_test)  # usually only out by 0.0005

    # shows no correlation between components
    # X_pca = pd.DataFrame(X_pca)
    # fig = plt.figure(figsize=(10, 8))
    # sns.heatmap(X_pca.corr(), annot=True)

    return model, X_train, X_test, y_test, pred


def train_k_fold(df, k):
    X = df[['open', 'high', 'low', 'close', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)
    y = df[['close']].values.astype(np.float32)

    # standardises the data - will not do much as data is already similar in size

    model = LinearRegression()

    kf = KFold(k)

    fold = 1
    best_score = 1
    worst_score = 0
    average_score = 0
    scores = []
    for train_index, validate_index in kf.split(X, y):
        model.fit(X[train_index], y[train_index])
        y_test = y[validate_index]
        pred = model.predict(X[validate_index])
        score = np.sqrt(metrics.mean_squared_error(pred, y_test))
        print(f"Fold:  #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(y[validate_index])}")
        print("Mean Squared error: {}".format(score))
        if score < best_score:
            best_score = score
        fold += 1
        if score > worst_score:
            worst_score = score
        average_score = (average_score + score) / 2

        scores.append(score)

        plt.figure(figsize=(15, 5))

        plt.plot(1, 2, 1)
        plt.plot(np.array(y_test[0:20]))
        plt.plot(pred[0:20])

        plt.title('close values')
        plt.xlabel('close')

    plt.figure(figsize=(15, 5))
    plt.plot(np.array(scores))
    plt.title('Mean Squared Averages Per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Score')

    print(f"Best Score: {best_score}")
    print(f"Worst Score: {worst_score}")
    print(f"Average Score: {average_score}")


def train(df):
    X = df[['open', 'high', 'low', 'close', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)
    y = df[['close']].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # standardises the data - will not do much as data is already similar in size

    model = LinearRegression()
    model.fit(X_train, y_train)

    # fitting and testing the model
    pred = model.predict(X_test)  # usually only out by 0.0005

    # shows correlation between the first 4 components
    # X = pd.DataFrame(X)
    # fig = plt.figure(figsize=(10, 8))
    # sns.heatmap(X.corr(), annot=True)

    return model, X_train, X_test, y_test, pred


def output(y_test, pred):
    score = np.sqrt(metrics.mean_squared_error(pred, y_test))

    # code for plotting the predicted stock data against actual stock data
    plt.figure(figsize=(15, 5))

    plt.plot(1, 2, 1)
    plt.plot(np.array(y_test[0:20]))
    plt.plot(pred[0:20])

    plt.title('close values')
    plt.xlabel('close')

    print("Mean Squared error: {}".format(score))


def correlation_test():
    dataframe = read('GOOG.csv')  # returns dataframe
    # heatmap(dataframe) #adjvolume and volume are not corrolated to other covariates
    pca(dataframe)  # only need the first component to capture almost 100% of the data


def runModel():
    dataframe = read('GOOG.csv')  # returns dataframe
    dataframe = shuffling(dataframe)  # returns dataframe
    trained_model = train(dataframe)  # return model, X_train, X_test, y_test, pred
    output(trained_model[3], trained_model[4])


def run_PCA_Model():
    dataframe = read('GOOG.csv')  # returns dataframe
    dataframe = shuffling(dataframe)  # returns dataframe
    trained_model = trained_pca(dataframe)  # return model, X_train, X_test, y_test, pred
    output(trained_model[3], trained_model[4])


def run_K_Fold_model():
    dataframe = read('GOOG.csv')  # returns dataframe
    dataframe = shuffling(dataframe)  # returns dataframe
    train_k_fold(dataframe, 10)  # return model, X_train, X_test, y_test, pred


if __name__ == "__main__":
    # correlation_test()
    # runModel()
    # run_PCA_Model()  # has a higher R squared value but no corrleation, suggesting hidden variables???
    run_K_Fold_model()
    plt.show()
