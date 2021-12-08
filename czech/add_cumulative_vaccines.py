import pandas as pd
import numpy as np



def main():
    df = pd.read_csv('vaccines.csv')

    df['date'] = df['date'].astype(np.datetime64)

    df['first_vaccine_cumulative'] = df['first_vaccine'].cumsum()
    df['second_vaccine_cumulative'] = df['second_vaccine'].cumsum()

    df.to_csv('vaccines.csv', index=False)

if __name__ == '__main__':
    main()