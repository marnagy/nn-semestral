import json
import matplotlib.pyplot as plt
from numpy.lib.function_base import diff
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

# scikit-learn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from argparse import Namespace, ArgumentParser
import pickle
from sys import stderr

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument('-a', '--absolute', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('-e', '--epsilon', type=float, default=.25)
    parser.add_argument('-d', '--difference', type=float, default=10)
    parser.add_argument('-i', '--iter', type=int, default=50_000)
    parser.add_argument('--step', type=int, default=500)

    args = parser.parse_args()

    if not ( 0 < args.epsilon < 1 ):
        print(f'Epsilon value has to be in interval (0,1). Given: {args.epsilon}', file=stderr)
        exit(1)
    
    if not ( args.difference > 0 ):
        print(f'Difference has to be a positive real number. Given: {args.difference}', file=stderr)
        exit(1)

    return args

def custom_cond(y, p, use_absolute: bool, eps: float, difference: float):
    if use_absolute:
        return abs(p - y) <= difference
    return (y == 0 and not p <= 5) or not (y * (1 - eps) <= p <= y * (1 + eps))

def train_by_step(X_train, Y_train, X_test, Y_test: DataFrame, norm_params, step_size, model: MLPRegressor):
    norm_ps = np.array([ x['p'] for x in norm_params ])
    norm_qs = np.array([ x['q'] for x in norm_params ])
    max_iter = model.max_iter

    unnorm_Y_test = norm_ps * Y_test + norm_qs
    model.set_params(max_iter=step_size)
    for i in range(max_iter // step_size):
        if i > 0:
            model.set_params(warm_start=True)

        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        unnorm_Y_pred = (norm_ps * Y_pred + norm_qs) #.reshape(unnorm_Y_test.shape)

        #print(unnorm_Y_test.shape, unnorm_Y_pred.shape)

        #print(f'Epoch {i*step_size}:')
        print(f'{i*step_size} {np.mean(np.abs(unnorm_Y_pred - unnorm_Y_test), axis=0)}')

def main():
    args = get_args()
    seed = args.seed
    X = pd.read_csv('inputs.csv')
    Y = pd.read_csv('outputs.csv')

    with open('outputs_norm_params.json', 'r', encoding='utf-8') as norm_params_file:
        output_normalization_params: list[tuple[float, float]] = json.loads( norm_params_file.read() )
    
    norm_ps = np.array([ x['p'] for x in output_normalization_params ])
    norm_qs = np.array([ x['q'] for x in output_normalization_params ])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=seed)
    Y_train, Y_test = Y_train.to_numpy().ravel(), Y_test.to_numpy().ravel()

    # print(len(X_train))
    # print(len(Y_train))
    # print(len(X_test))
    # print(len(Y_test))


    model = MLPRegressor(hidden_layer_sizes=(1_000), activation='tanh',
        solver='adam', learning_rate='adaptive',
        alpha=.001,
        tol=0.000001, n_iter_no_change=args.iter,
        max_iter=args.iter, random_state=seed,
        #verbose=True
        )
    train_by_step(X_train, Y_train, X_test, Y_test, output_normalization_params, 500, model)
    #model.fit()

    #Y_pred = model.predict(X_test)

    # for pred, (index, row_test) in zip(Y_pred, Y_test.iterrows()):
    #     print(f'Prediction: {pred}')
    #     print(f'Actual: {row_test.to_numpy()}')
    # exit()

    # "unnormalize"
    # Y_pred = norm_ps * Y_pred + norm_qs
    # Y_test = norm_ps * Y_test + norm_qs

    # print(np.sum( np.abs( Y_pred - Y_test ), axis=0 ))

    # eps = args.epsilon # allowed percent of error
    # with open('train_compare.log', 'w') as comp_file:
    #     accuracies = list()
    #     for pred, (_, row_series) in zip(Y_pred, Y_test.iterrows()):
    #         row = row_series.to_numpy()
    #         err = list()
    #         print(f'Pred: {pred} Actual: {row}', file=comp_file)
    #         for _,(p, y) in enumerate(zip(pred, row)):
    #             y = int(y)
    #             if custom_cond(y,p, args.absolute, eps, args.difference):
    #                 err.append( abs(y - p) )
    #             else:
    #                 err.append(0)
    #         accuracies.append(err)
    
    # accuracies = np.array(accuracies)

    # print(accuracies, file=stderr)
    
    # if args.absolute:
    #     print(f'Average accuracy within difference {args.difference} across {Y_test.shape[0]} inputs:', file=stderr)
    # else:
    #     print(f'Average accuracy with eps {eps} accross {Y_test.shape[0]} inputs:', file=stderr)
    # print(np.mean(accuracies, axis=0), file=stderr)

    with open('model.pickle', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    #return Y_pred, Y_test

if __name__ == '__main__':
    main()