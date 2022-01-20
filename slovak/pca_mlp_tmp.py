

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
import sys
from sys import stderr
import os

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

def main():
    dirname = os.path.dirname(__file__)
    seed = 42
    X = pd.read_csv(dirname+'/inputs_norm.csv')
    Y = pd.read_csv(dirname+'/outputs_norm.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9, random_state=seed)
    pca = PCA(random_state=seed)

    mlp = MLPRegressor()  #(hidden_layer_sizes=(1_000), activation='tanh',
    #     solver='adam', learning_rate='adaptive', max_iter=1_000, alpha=0.9,
    #     momentum=0.2, random_state=seed,
    #     verbose=True, tol=0.000001, n_iter_no_change=100)


    pipe = Pipeline(steps=[("pca", pca), ("regressor", mlp)])


#{', 'regressor__learning_rate': 'adaptive', 'regressor__max_iter': 100, 'regressor__momentum': 0.1, 'regressor__random_state': 42, 'regressor__solver': 'adam'}


    # param_grid = dict(pca=['passthrough', PCA(25), PCA(42), PCA(60)],
    #     regressor__hidden_layer_sizes= [(500),(500,2),(500,3)], 
    #     regressor__activation = ["tanh"], 
    #     regressor__solver = ["adam", "sgd"], 
    #     regressor__learning_rate = ['adaptive'], 
    #     regressor__max_iter = [75, 100, 125, 150], 
    #     regressor__alpha = [0.05, 0.1, 0.2, 0.5, 0.9],
    #     regressor__momentum = [0.05, 0.1, 0.2],
    #     regressor__random_state = [seed])
    # param_grid = {
    #     "pca__n_components": [25, 42, 60],
    #     "regressor__hidden_layer_sizes": [(100),(100,2),(100,3),(500),(1_000)], 
    #     "regressor__activation": ["tanh"], 
    #     "regressor__solver": ["adam", "sgd"], 
    #     "regressor__learning_rate": ['adaptive'], 
    #     "regressor__max_iter": [100, 350, 700, 1000], 
    #     "regressor__alpha": [0.0001, 0.001, 0.01, 0.1],
    #     "regressor__momentum": [0.1, 0.5, 0.9],
    #     "regressor__random_state": [seed],
    #     #tol=0.000001, n_iter_no_change=100 ??
    # }
    param_grid = {
        "pca__n_components": np.linspace(20, 35, num=4, dtype=int),
        "regressor__hidden_layer_sizes": [(550),(575),(600),(625),(650)], 
        "regressor__activation": ["tanh"], 
        "regressor__solver": ["adam"], 
        "regressor__learning_rate": ['adaptive'], 
        "regressor__max_iter": np.linspace(45, 76, num=16, dtype=int), 
        "regressor__alpha": [0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11],
        "regressor__random_state": [seed],
        #tol=0.000001, n_iter_no_change=100 ??
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=1)
    search.fit(X_train, Y_train)

    with open('search_tmp7.pickle', 'wb') as model_file:
        pickle.dump(search, model_file)

    print(search.best_params_)
    return

    
    pca_X_test = search.best_estimator_["pca"].transform(X_test)
    Y_pred = search.predict(pca_X_test)

    eps = 0.05 # allowed percent of error
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
    
    print("PCA+MLP")
    print(f'Average error with eps {eps}: {sum(accuracies) / len(accuracies) :.4f}', file=stderr)
    print("RMSE:  ")
    print(mean_squared_error(Y_test, Y_pred, squared=False))

    

    #   60-0.03957094124639943     42-0.04029050204022203    25-0.03888817451936788        16-0.03954949206088276

    # print(pca.explained_variance_ratio_)
    # print(pca.n_features_)
    # return

if __name__ == '__main__':
    main()