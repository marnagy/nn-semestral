import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import pickle
from sys import stderr
import os

def accuracy_with_tolerance(Y_pred, Y_test):
    eps = 0.5 # allowed percent of error
    assert 0 < eps < 1
    accuracies = list()
    for pred, (row_index, row_series) in zip(Y_pred, Y_test.iterrows()):
        row = row_series.to_numpy()
        err = 0
        #print(f'Pred: {pred} Actual: {row}', file=stderr)
        for i,(p, y) in enumerate(zip(pred, row)):
            #print(p, y)
            y = int(y)
            # if y == 0:
            #     if not (p <= 5):
            #         err += abs(y - p)
            if (y == 0 and not p <= 5) or not (y * (1 - eps) <= p <= y * (1 + eps)):
                err += abs(y - p)
        accuracies.append(err)
    return sum(accuracies) / len(accuracies)

def main():
    dirname = os.path.dirname(__file__)
    seed = 42
    X = pd.read_csv(dirname + '/inputs.csv')
    Y = pd.read_csv(dirname + '/outputs.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9, random_state=seed)
    pca = PCA(n_components=30)

    mlp = MLPRegressor(hidden_layer_sizes=(100), activation='tanh',
         solver='adam', learning_rate='adaptive', max_iter=1_000, alpha=0.2,
         momentum=0.05, random_state=seed,
         verbose=False, tol=0.000001, n_iter_no_change=100)


    pipe = Pipeline(steps=[("pca", pca), ("regressor", mlp)])



    # param_grid = {
    #     "pca__n_components": [12, 20, 40],
    #     #"pca__random_state": [seed],
    #     "regressor__hidden_layer_sizes": [(100),(500),(1_000),(2_000)], 
    #     "regressor__activation": ["logistic", "tanh", "relu"], 
    #     "regressor__solver": ["adam", "sgd"], 
    #     "regressor__learning_rate": ['adaptive'], 
    #     "regressor__max_iter": [100, 500], #, 1000, 2000, 5000], 
    #     "regressor__alpha": [0.005, 0.5, 0.9],
    #     "regressor__momentum": [0.2, 0.5, 0.9],
    #     "regressor__random_state": [seed],
    #     #tol=0.000001, n_iter_no_change=100 ??
    # }
    # search = GridSearchCV(pipe, param_grid, n_jobs=4, verbose=2)
    pipe.fit(X_train, Y_train)

    with open('manual_test_model.pkl', 'wb') as model_file:
        pickle.dump(pipe, model_file)

    Y_pred = pipe.predict(X_test)

    acc_with_tolerance = accuracy_with_tolerance(Y_pred, Y_test)
    
    print("PCA+MLP")
    print(f'Average error with eps 10%: {acc_with_tolerance}', file=stderr)
    print("RMSE:  ")
    print(mean_squared_error(Y_test, Y_pred, squared=False))

    #   60-0.03957094124639943     42-0.04029050204022203    25-0.03888817451936788        16-0.03954949206088276

    # print(pca.explained_variance_ratio_)
    # print(pca.n_features_)
    # return

if __name__ == '__main__':
    main()