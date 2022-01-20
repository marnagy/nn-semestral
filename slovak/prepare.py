# Marekov kod
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from argparse import Namespace, ArgumentParser

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('-f', '--file', required=False)
    parser.add_argument('--before', type=int, default=10)
    parser.add_argument('--future', type=int, default=2)
    parser.add_argument('-n', '--normalize', type=bool, default=True, nargs='?', const=True)

    return parser.parse_args()

def normalize(series: pd.Series, start: int = 0, end: int = 1) -> pd.Series:
    series.dtypes
    size = series.max() - series.min()
    print("p: ", size, "q: ", series.min())
    return ( series - series.min() ) / size
    kumulativni_pocet_nakazenych
# p:  1207727.0 q:  1.0
# prirustkovy_pocet_umrti
# p:  14696.0 q:  0.0
# kumulativni_pocet_umrti
# p:  3685408.0 q:  0.0
# first_vaccine_cumulative
# p:  2651767.0 q:  0.0
# second_vaccine_cumulative
# p:  2349713.0 q:  0.0
# prirustkovy_pocet_nakazenych
# p:  15289.0 q:  0.0
# prirustkovy_pocet_umrti
# p:  14696.0 q:  0.0

def main():
    args = get_args()
    df = pd.read_csv('new_combined.csv')

    # # add column 'currently_sick'
    # df['currently_sick'] = df['kumulativni_pocet_nakazenych'] - (df['kumulativni_pocet_vylecenych'] + df['kumulativni_pocet_umrti'])

    input_columns = [
        'prirustkovy_pocet_nakazenych',
        'kumulativni_pocet_nakazenych',
        # 'prirustkovy_pocet_vylecenych',
        # 'kumulativni_pocet_vylecenych',
        'prirustkovy_pocet_umrti',
        'kumulativni_pocet_umrti',
        'first_vaccine_cumulative',
        'second_vaccine_cumulative',
        # 'currently_sick'
    ]
    output_columns = [
        # 'currently_sick',
        'prirustkovy_pocet_nakazenych',
        # 'prirustkovy_pocet_vylecenych',
        'prirustkovy_pocet_umrti'
    ]
    inputs_df = df[input_columns]
    outputs_df = df[output_columns]

    if args.normalize:
        print('Normalizing')
        for curr_df in [inputs_df, outputs_df]:
            # print(curr_df.columns)
            # columns_to_scale = list(filter(lambda type: np.isscalar, curr_df.columns))
            # print(columns_to_scale)
            for col in curr_df.columns:
                #print(col)
                curr_df[col] = curr_df[col].astype(np.float64)
                print(col)
                curr_df[col] = normalize(curr_df[col])

    inputs = np.empty(shape=(0, len(input_columns) * args.before))
    outputs = np.empty(shape=(0, len(output_columns) * args.future))
    for index in range(args.before - 1, len(df) - args.future):
        in_arr = np.empty(shape=(0))
        out_arr = np.empty(shape=(0))

        step_start = index - args.before + 1
        step_end = index + args.future
        for i in range(step_start, step_end + 1):
            if i <= index: # before part
                in_arr = np.hstack((in_arr, inputs_df.iloc[i].to_numpy())) 
            else: # future part
                out_arr = np.hstack((out_arr, outputs_df.iloc[i].to_numpy()))
        # print(f'in_arr: {in_arr}')
        # print(f'out_arr: {out_arr}')

        # print(f'in_arr shape: {in_arr.shape}')
        # print(f'out_arr shape: {out_arr.shape}')
        
        inputs = np.vstack((inputs, in_arr))
        outputs = np.vstack((outputs, out_arr))
    
    print(f'Inputs shape: {inputs.shape}')
    print(f'Outputs shape: {outputs.shape}')

    inputs_df = pd.DataFrame(inputs)
    inputs_df.to_csv('new_inputs.csv', index=False)

    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv('new_outputs.csv', index=False)


if __name__ == '__main__':
    main()