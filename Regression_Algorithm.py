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
from joblib import dump, load


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
    # np.random.seed(42)  # Uncomment this line to get the same shuffle each time
    df = df.reindex(np.random.permutation(df.index))
    df.reset_index(inplace=True, drop=True)
    return df

    # shuffling seems either not do much or greatly improve the result
    # could indicate the use of kfold split to get the best model

    # null_columns=df.columns[df.isnull().any()] #Contains 0 null columns


def pca(df):
    # if possible to auto generate x,y with all features instead of manually picking features for x
    x = df[['open', 'high', 'low', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)

    sc = StandardScaler()
    
    x_scaled = sc.fit_transform(x)

    # pca = PCA(n_components=None) #shows that only need 1 component to capture 100% of the data
    pca_bot = PCA(n_components=None)
    pca_bot.fit(x_scaled)

    # Get the eigenvalues
    print("Eigenvalues:")
    print(pca_bot.explained_variance_)
    print()

    # Get explained variances
    print("Variances (Percentage):")
    print(pca_bot.explained_variance_ratio_ * 100)
    print()

    # Make the scree plot
    plt.plot(np.cumsum(pca_bot.explained_variance_ratio_ * 100))
    plt.xlabel("Number of components (Dimensions)")
    plt.ylabel("Explained variance (%)")


def trained_pca(df):
    x = df[['open', 'high', 'low', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)
    y = df[['close']].values.astype(np.float32)

    sc = StandardScaler()
    x_scaled = sc.fit_transform(x)
    pca_bot = PCA(n_components=2)
    x_pca = pca_bot.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

    # standardises the data - will not do much as data is already similar in size

    model = LinearRegression()
    model.fit(x_train, y_train)

    # fitting and testing the model
    pred = model.predict(x_test)  # usually only out by 0.0005

    dump(model, "pca_train.joblib")

    # shows no correlation between components
    # x_pca = pd.DataFrame(x_pca)
    # fig = plt.figure(figsize=(10, 8))
    # sns.heatmap(x_pca.corr(), annot=True)

    return model, x_train, x_test, y_test, pred


def train_k_fold(df, k):

    x = df[['open', 'high', 'low', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)
    y = df[['close']].values.astype(np.float32)

    model = LinearRegression()

    # trains with k fold where k is deternimed by the parameter k, outputs prediction graphs for each fold and error
    # then produces a graph of error per fold as well as a best,worst and average error

    kf = KFold(k)

    fold = 1
    best_score = 0
    worst_score = 0
    average_score = 0
    scores = []
    for train_index, validate_index in kf.split(x, y):
        model.fit(x[train_index], y[train_index])
        y_test = y[validate_index]
        pred = model.predict(x[validate_index])
        score = np.sqrt(metrics.mean_squared_error(pred, y_test))
        print(f"Fold:  #{fold}, Training Size: {len(x[train_index])}, Validation Size: {len(y[validate_index])}")
        print("Mean Squared error: {}".format(score))
        if (score < best_score) or (best_score == 0):
            best_score = score
            dump(model, "k_fold_best.joblib")
        fold += 1
        if score > worst_score:
            worst_score = score
        if average_score == 0:
            average_score = score
        else:
            average_score = (average_score + score) / 2

        scores.append(score)

        plt.figure(figsize=(15, 5))

        plt.plot(1, 2, 1)
        plt.plot(np.array(y_test[0:20]))
        plt.plot(pred[0:20])

        plt.title('close values')
        plt.xlabel('close')

    plt.figure(figsize=(15, 5))
    plt.bar([n for n in range(k)], scores)
    plt.title('Mean Squared Averages Per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Score')

    print(f"Best Score: {best_score}")
    print(f"Worst Score: {worst_score}")
    print(f"Average Score: {average_score}")


def train_k_fold_pca(df, k):

    df_s = shuffling(df)

    x = df_s[['open', 'high', 'low', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)
    y = df_s[['close']].values.astype(np.float32)

    sc = StandardScaler()
    pca_bot = PCA(n_components=2)
    x_scaled = sc.fit_transform(x)
    x_pca = pca_bot.fit_transform(x_scaled)

    model = LinearRegression()

    # trains with k fold where k is determined by the parameter k, outputs prediction graphs for each fold and error
    # then produces a graph of error per fold as well as a best,worst and average error

    kf = KFold(k)

    fold = 1
    best_score = 0
    worst_score = 0
    average_score = 0
    scores = []
    for train_index, validate_index in kf.split(x_pca, y):
        model.fit(x_pca[train_index], y[train_index])
        y_test = y[validate_index]
        pred = model.predict(x_pca[validate_index])
        score = np.sqrt(metrics.mean_squared_error(pred, y_test))
        print(f"Fold:  #{fold}, Training Size: {len(x_pca[train_index])}, Validation Size: {len(y[validate_index])}")
        print("Mean Squared error: {}".format(score))
        if (score < best_score) or (best_score == 0):
            best_score = score
            dump(model, "k_fold_pca_best.joblib")
        fold += 1
        if score > worst_score:
            worst_score = score
        if average_score == 0:
            average_score = score
        else:
            average_score = (average_score + score) / 2

        scores.append(score)

        plt.figure(figsize=(15, 5))

        plt.plot(1, 2, 1)
        plt.plot(np.array(y_test[0:20]))
        plt.plot(pred[0:20])

        plt.title('close values')
        plt.xlabel('close')

    plt.figure(figsize=(15, 5))
    plt.bar([n for n in range(k)], scores)
    plt.title('Mean Squared Averages Per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Score')

    print(f"Best Score: {best_score}")
    print(f"Worst Score: {worst_score}")
    print(f"Average Score: {average_score}")


def train(df):
    x = df[['open', 'high', 'low', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)
    y = df[['close']].values.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # standardises the data - will not do much as data is already similar in size

    model = LinearRegression()
    model.fit(x_train, y_train)

    # fitting and testing the model
    pred = model.predict(x_test)  # usually only out by 0.0005

    dump(model, "basic_train.joblib")

    # shows correlation between the first 4 components
    # x = pd.DataFrame(x)
    # fig = plt.figure(figsize=(10, 8))
    # sns.heatmap(x.corr(), annot=True)

    return model, x_train, x_test, y_test, pred


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
    # heatmap(dataframe) #adjvolume and volume are not correlated to other co-variates
    pca(dataframe)  # only need the first component to capture almost 100% of the data


def run_model():
    dataframe = read('GOOG.csv')  # returns dataframe
    dataframe = shuffling(dataframe)  # returns dataframe
    trained_model = train(dataframe[0:1007])  # return model, x_train, x_test, y_test, pred
    output(trained_model[3], trained_model[4])


def run_pca_model():
    dataframe = read('GOOG.csv')  # returns dataframe
    dataframe = shuffling(dataframe)  # returns dataframe
    trained_model = trained_pca(dataframe[0:1007])  # return model, x_train, x_test, y_test, pred
    output(trained_model[3], trained_model[4])


def run_k_fold_model():
    dataframe = read('GOOG.csv')  # returns dataframe
    dataframe = shuffling(dataframe[0:1007])  # returns dataframe
    train_k_fold(dataframe, 10)  # output graph of errors for k fold analysis k=10


def run_k_fold_model_pca():
    dataframe = read('GOOG.csv')  # returns dataframe
    dataframe = shuffling(dataframe[0:1007])  # returns dataframe
    train_k_fold_pca(dataframe, 10)  # output graph of errors for k fold analysis k=10 using pca


def plot_stocks_against_pred():
    dataframe = read('GOOG.csv')  # returns dataframe
    dataframe = dataframe[1008:1258]
    x = dataframe[['open', 'high', 'low', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)
    y = dataframe[['close']].values.astype(np.float32)

    filename = input("Please input model filename (filename.joblib)")

    model = load(filename)

    pred = model.predict(x)

    plt.figure(1, figsize=(20, 20))

    x_ax = [n for n in range(1, 251)]

    plt.plot(x_ax, pred, label="predictions")
    plt.plot(x_ax, y, label="actual values")

    plt.legend()

    plt.title('Prediction evaluation')
    plt.ylabel('close')

    print(np.sqrt(metrics.mean_squared_error(pred, y)))


def plot_stocks_against_pred_pca():
    dataframe = read('GOOG.csv')  # returns dataframe
    dataframe = dataframe[1008:1258]
    x = dataframe[['open', 'high', 'low', 'volume', 'divCash', 'splitFactor']].values.astype(np.float32)
    y = dataframe[['close']].values.astype(np.float32)

    sc = StandardScaler()
    pca_bot = PCA(n_components=2)
    x_scaled = sc.fit_transform(x)
    x_pca = pca_bot.fit_transform(x_scaled)

    filename = input("Please input model filename (filename.joblib)")

    model = load(filename)

    pred = model.predict(x_pca)

    plt.figure(1, figsize=(10, 10))

    x_ax = [n for n in range(1, 251)]

    plt.plot(x_ax, pred, label="predictions")
    plt.plot(x_ax, y, label="actual values")

    plt.legend()

    plt.title('Prediction evaluation')
    plt.ylabel('close')

    print(np.sqrt(metrics.mean_squared_error(pred, y)))


if __name__ == "__main__":
    # correlation_test()
    # run_model()
    # run_pca_model()  # has a higher R squared value but no correlation, suggesting hidden variables???
    # run_k_fold_model_pca()
    # run_k_fold_model()
    plot_stocks_against_pred()
    # plot_stocks_against_pred_pca()
    plt.show()
