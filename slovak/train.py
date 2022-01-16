# Marekov kod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# scikit-learn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from argparse import Namespace, ArgumentParser
import pickle
from sys import stderr

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('-s', '--seed', type=int, default=42)

    return parser.parse_args()

def main():
    args = get_args()
    seed = args.seed
    X = pd.read_csv('inputs_norm.csv')
    Y = pd.read_csv('outputs_norm.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9, random_state=seed)

    # print(len(X_train))
    # print(len(Y_train))
    # print(len(X_test))
    # print(Y_test)

    model = MLPRegressor(hidden_layer_sizes=(1_000), activation='tanh',
        solver='adam', learning_rate='adaptive', max_iter=1_000, alpha=0.9,
        momentum=0.2, random_state=seed,
        verbose=True, tol=0.000001, n_iter_no_change=100)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    # for pred, (index, row_test) in zip(Y_pred, Y_test.iterrows()):
    #     print(f'Prediction: {pred}')
    #     print(f'Actual: {row_test.to_numpy()}')
    # exit()

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
    
    print("MLP")
    print(f'Average error with eps {eps}: {sum(accuracies) / len(accuracies) :.4f}', file=stderr)
    print("RMSE:  ")
    print(mean_squared_error(Y_test, Y_pred, squared=False))

    with open('model.pickle', 'wb') as model_file:
        pickle.dump(model, model_file)

if __name__ == '__main__':
    main()