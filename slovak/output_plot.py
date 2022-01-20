import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import datetime

import pickle
from sys import stderr

def main():
    seed = 42
    Yt = pd.read_csv('Yt.csv')
    Yp = pd.read_csv('Yp.csv')

    df = pd.read_csv('new_combined.csv')
    chory = df['prirustkovy_pocet_nakazenych'][-100:]
    # plt.plot(chory[-100:])

    # plt.show()

    # plt.figure
    mrtvy = df['prirustkovy_pocet_umrti'][-100:]
    # plt.plot(mrtvy[-100:])
    # plt.show()
    
    true_nakazeny = Yt.iloc[:, 0]
    true_mrtvy = Yt.iloc[:, 1]

    base = datetime.datetime(2021, 10, 10)
    arr = np.array([base + datetime.timedelta(days=i) for i in range(100)])
    # print(arr)
    # return

    # plt.plot(arr, true_nakazeny)
    # for i in range(44):
    #     plt.plot([arr[i], arr[i+1]],[Yp.iloc[i, 0], Yp.iloc[i, 2]])
    # plt.title('New sick')

    # plt.show()


    # plt.plot(arr, true_mrtvy)
    # for i in range(44):
    #     plt.plot([arr[i], arr[i+1]],[Yp.iloc[i, 1], Yp.iloc[i, 3]])
    # plt.title('Cummulative number of deaths')
    # plt.show()
    # # print(true_mrtvy)


    
    figure, axis = plt.subplots(1, 2)
    figure.set_figwidth(20)
    figure.set_figheight(10)

    axis[0].set_title('New sick')
    axis[0].plot(arr, chory)
    for i in range(44):
        axis[0].plot([arr[i+55], arr[i+55+1]],[Yp.iloc[i, 0], Yp.iloc[i, 2]])

    axis[1].set_title('Cummulative number of deaths')
    axis[1].plot(arr, mrtvy)
    for i in range(44):
        axis[1].plot([arr[i+55], arr[i+55+1]],[Yp.iloc[i, 1], Yp.iloc[i, 3]])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()