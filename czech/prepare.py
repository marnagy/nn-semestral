from argparse import ArgumentParser, Namespace
from os import spawnle
from sys import stderr

import pandas as pd
import numpy as np
from sklearn.externals._packaging.version import parse

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('-f', '--file', type=str, required=True, help='Path to CSV file.')
    #arser.add_argument('--datum-encode', default=False, nargs='?', const=True, help='Include date in input.')
    #parser.add_argument('--datum-onehot', default=False, nargs='?', const=True, help='Encode date in onehot encoding.')
    parser.add_argument('--before', type=int, default=5, help="How many days in past should script consider")
    parser.add_argument('--future', type=int, default=3, help="How many days in the future should script predict")
    parser.add_argument('-v', '--verbose', type=bool, default=False, nargs='?', const=True)

    args = parser.parse_args()

    if not args.file.endswith('.csv'):
        print(f'File {args.file} does not have correct format (csv)')
        exit(1)
    
    return args

def main():
    args = get_args()

    df = pd.read_csv(args.file) #, index_col='datum')
    df['datum'] = df['datum'].astype(np.datetime64)

    chosen_columns = [
        'datum',
        'prirustkovy_pocet_nakazenych',
        'kumulativni_pocet_nakazenych',
        'prirustkovy_pocet_vylecenych',
        'kumulativni_pocet_vylecenych',
        'prirustkovy_pocet_umrti',
        'kumulativni_pocet_umrti',
        #'prirustkovy_pocet_provedenych_testu',
        #'kumulativni_pocet_testu',
        #'prirustkovy_pocet_provedenych_ag_testu',
        #'kumulativni_pocet_ag_testu',
        'first_vaccine_cumulative',
        'second_vaccine_cumulative',
        #'stav_bez_priznaku',
        #'stav_lehky',
        #'stav_stredni',
        #'stav_tezky'
    ]
    input_columns = [
        'prirustkovy_pocet_nakazenych',
        #'kumulativni_pocet_nakazenych',
        'prirustkovy_pocet_vylecenych',
        #'kumulativni_pocet_vylecenych',
        'prirustkovy_pocet_umrti',
        #'kumulativni_pocet_umrti',
        'first_vaccine_cumulative',
        'second_vaccine_cumulative',
        'aktualne_nakazenych'
    ]
    output_columns = [
        'prirustkovy_pocet_nakazenych',
        #'kumulativni_pocet_nakazenych',
        #'prirustkovy_pocet_vylecenych',
        #'kumulativni_pocet_vylecenych',
        'prirustkovy_pocet_umrti',
        #'kumulativni_pocet_umrti',
    ]

    df = df[chosen_columns]
    df['aktualne_nakazenych'] = df['kumulativni_pocet_nakazenych'] - (df['kumulativni_pocet_vylecenych'] + df['kumulativni_pocet_umrti'])
    df['aktualne_nakazenych'] = df['aktualne_nakazenych'].astype(np.int32)

    # remove last row
    df = df.drop(len(df) - 1)

    # remove 'datum' column
    df = df.drop('datum', axis='columns')

    inputs_df = df[input_columns]
    outputs_df = df[output_columns]

    # if args.verbose:
    #     df.info()

    inputs_width_for_day = len(inputs_df.columns)
    outputs_width_for_day = len(outputs_df.columns)
    #patterns = np.empty(shape=(0, args.before * inputs_width_for_day + args.future * outputs_width_for_day))
    x = np.empty(shape=(0, args.before * inputs_width_for_day))
    y = np.empty(shape=(0, args.future * outputs_width_for_day))

    indices = list(df.index)
    split_index = args.before * inputs_width_for_day
    for index in range(args.before - 1, len(indices) - args.future - 1):
        arr = np.empty(shape=(0,))
        for i in range(index - args.before + 1, index + args.future + 1):
            if i <= index:
                arr = np.hstack((arr, inputs_df.iloc[i].to_numpy()))
            else:
                arr = np.hstack((arr, outputs_df.loc[i].to_numpy()))
            #print(arr.shape)
        #patterns = np.vstack((patterns, arr))
        x = np.vstack((x, arr[:split_index]))
        y = np.vstack((y, arr[split_index:]))
    
    if args.verbose:
        #print(patterns.shape)
        print(x.shape)
        print(y.shape)

    #x.tofile('inputs.csv')
    in_columns = list()
    for i in range(args.before, 0, -1):
        for col in inputs_df.columns:
            in_columns.append(f'{col}-{i}')
    inputs_df = pd.DataFrame(x, columns=in_columns)
    for col in inputs_df.columns:
        inputs_df[col] = inputs_df[col].astype(np.int32)
    inputs_df.to_csv('inputs.csv', index=False)
    #y.tofile('outputs.csv')
    out_columns = list()
    for i in range(args.future):
        for col in outputs_df.columns:
            out_columns.append(f'{col}-{i+1}')
    outputs_df = pd.DataFrame(y, columns=out_columns)
    for col in outputs_df.columns:
        outputs_df[col] = outputs_df[col].astype(np.int32)
    outputs_df.to_csv('outputs.csv', index=False)

if __name__ == '__main__':
    main()