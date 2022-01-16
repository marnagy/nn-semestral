import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import pickle
from sys import stderr

def main():
    seed = 42
    X = pd.read_csv('inputs_norm.csv')
    Y = pd.read_csv('outputs_norm.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9, random_state=seed)

    reg = LinearRegression().fit(X_train, Y_train)
    print(reg.score(X_train, Y_train))

    Y_pred = reg.predict(X_test)

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
    
    print(f'linear regression..... Average error with eps {eps}: {sum(accuracies) / len(accuracies) :.4f}', file=stderr)

    with open('reg.pickle', 'wb') as model_file:
        pickle.dump(reg, model_file)




    # clf = LogisticRegression(random_state=0).fit(X_train, Y_train)

    # Y_pred = clf.predict(X_test)

    # eps = 0.05 # allowed percent of error
    # accuracies = list()
    # for pred, (row_index, row_series) in zip(Y_pred, Y_test.iterrows()):
    #     row = row_series.to_numpy()
    #     err = 0
    #     #print(f'Pred: {pred} Actual: {row}', file=stderr)
    #     for i,(p, y) in enumerate(zip(pred, row)):
    #         #print(p, y)
    #         y = int(y)
    #         # if y == 0:
    #         #     if not (p <= 5):
    #         #         err += abs(y - p)
    #         if (y == 0 and not p <= 5) or not (y * (1 - eps) <= p <= y * (1 + eps)):
    #             err += abs(y - p)
    #     accuracies.append(err)
    
    # print(f'logistic...... Average error with eps {eps}: {sum(accuracies) / len(accuracies) :.4f}', file=stderr)

    # with open('clf.pickle', 'wb') as model_file:
    #     pickle.dump(clf, model_file)


if __name__ == '__main__':
    main()