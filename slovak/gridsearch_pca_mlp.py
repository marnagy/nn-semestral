import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

import pickle
from sys import stderr

def main():
    seed = 42
    X = pd.read_csv('inputs_norm.csv')
    Y = pd.read_csv('outputs_norm.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9, random_state=seed)
    regressor = MLPRegressor(hidden_layer_sizes=(1_000), activation='tanh',
        solver='adam', learning_rate='adaptive', max_iter=1_000, alpha=0.1,
        momentum=0.2, random_state=seed,
        verbose=True, tol=0.000001, n_iter_no_change=100)
    model = Pipeline([('pca', PCA(20, random_state=seed)), ('MLPRegressor', regressor)])
    # pca = PCA()#n_components=16)
    # pca_X_train = pca.fit_transform(X_train)
    # pca_X_test = pca.transform(X_test)

    # 60-0.03957094124639943     42-0.04029050204022203    25-0.03888817451936788        16-0.03954949206088276

    # PCA(15) -> MLPRegressor(hidden_layer_sizes=5000)
    # RMSE: 0.040198700534374646

    # PCA(15) -> MLPRegressor(hidden_layer_sizes=1000, max_iter=100)
    # RMSE: 0.03492254892524389

    # PCA(20) -> MLPRegressor(hidden_layer_sizes=1000, max_iter=100)
    # RMSE: 0.034780740237713055

    # PCA(20) -> MLPRegressor(hidden_layer_sizes=1000, max_iter=1000, alpha=0.1, n_iter_no_change=100)
    # RMSE: 0.030511733884162368

    # PCA(20) -> MLPRegressor(hidden_layer_sizes=5000)
    # RMSE: 0.038870081729906664

    # PCA(20) -> MLPRegressor(hidden_layer_sizes=5000, max_iter=5_000)
    # RMSE: 0.03879691401176465

    # PCA(30) -> MLPRegressor(hidden_layer_sizes=5000)
    # RMSE: 0.04050308466146327

    # print(pca.explained_variance_ratio_)
    # print(pca.n_features_)
    # return

    # model = MLPRegressor(hidden_layer_sizes=(1_000), activation='tanh',
    #     solver='adam', learning_rate='adaptive', max_iter=1_000, alpha=0.9,
    #     momentum=0.2, random_state=seed,
    #     verbose=True, tol=0.000001, n_iter_no_change=100)
    # model.fit(pca_X_train, Y_train)

    # Y_pred = model.predict(pca_X_test)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    eps = 0.05 # allowed percent of error
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
    
    print("PCA+MLP")
    print(f'Average error with eps {eps}: {sum(accuracies) / len(accuracies) :.4f}', file=stderr)
    print("RMSE:  ")
    print(mean_squared_error(Y_test, Y_pred, squared=False))

if __name__ == '__main__':
    main()