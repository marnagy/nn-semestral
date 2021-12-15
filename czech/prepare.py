import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from argparse import Namespace, ArgumentParser

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('--before', type=int, default=10)
    parser.add_argument('--future', type=int, default=2)

    return parser.parse_args()

def main():
    args = get_args()
    df = pd.read_csv('combined.csv')

    # remove last row (incomplete data)
    df = df.drop(len(df['datum']) - 1)

    # add column 'currently_sick'
    df['currently_sick'] = df['kumulativni_pocet_nakazenych'] - (df['kumulativni_pocet_vylecenych'] + df['kumulativni_pocet_umrti'])

    input_columns = [
        'prirustkovy_pocet_nakazenych',
        'kumulativni_pocet_nakazenych',
        'prirustkovy_pocet_vylecenych',
        'kumulativni_pocet_vylecenych',
        'prirustkovy_pocet_umrti',
        'kumulativni_pocet_umrti',
        'first_vaccine_cumulative',
        'second_vaccine_cumulative',
        'currently_sick'
    ]
    output_columns = [
        'currently_sick',
        'prirustkovy_pocet_nakazenych',
        'prirustkovy_pocet_vylecenych',
        'prirustkovy_pocet_umrti'
    ]
    inputs_df = df[input_columns]
    outputs_df = df[output_columns]

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

    inputs_df = pd.DataFrame(inputs, dtype=np.int32)
    inputs_df.to_csv('inputs.csv')

    outputs_df = pd.DataFrame(outputs, dtype=np.int32)
    outputs_df.to_csv('outputs.csv')


if __name__ == '__main__':
    main()