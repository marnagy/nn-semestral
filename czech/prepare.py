from argparse import Namespace, ArgumentParser
import json
import numpy as np
import pandas as pd

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('--before', type=int, default=10)
    parser.add_argument('--future', type=int, default=2)
    parser.add_argument('-n', '--normalize', type=bool, default=False, nargs='?', const=True)

    return parser.parse_args()

def normalize(series: pd.Series) -> tuple[pd.Series, float, float]:
    series.dtypes
    size = series.max() # - series.min()
    q = 0 #series.min()
    p = size
    return ( series - q ) / p, p, q

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
        #'currently_sick',
        'prirustkovy_pocet_nakazenych',
        #'prirustkovy_pocet_vylecenych',
        'prirustkovy_pocet_umrti'
    ]

    row_indices_to_remove = list()
    for row_index, row in df.iterrows():
        if row['kumulativni_pocet_nakazenych'] == 0:
            row_indices_to_remove.append( row_index )
    
    df = df.drop(index=row_indices_to_remove)
    print(f'Dropped { len(row_indices_to_remove) } rows.')
    del row_indices_to_remove

    d = dict()
    if args.normalize:
        print('Normalizing')
        for col_name in df.columns:
            #print(col)
            if df[col_name].dtype != np.int64:
                continue

            #print(f'Normalizing column: {col_name}')
            df[col_name] = df[col_name].astype(np.float64)
            df[col_name], p, q = normalize(df[col_name])
            d[col_name] = { 'p': p, 'q': q }
        
        # with open('normalization_params.json', 'w') as norm_param_file:
        #     json.dump(d, indent=2, fp=norm_param_file, )

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

    inputs_df = pd.DataFrame(inputs)
    inputs_df.to_csv('inputs.csv', index=False)

    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv('outputs.csv', index=False)

    if args.normalize:
        params = list()
        for _ in range(args.future):
            for col_name in output_columns:
                params.append( d[col_name] )

        with open('outputs_norm_params.json', 'w') as norm_params_file:
            json.dump(params, fp=norm_params_file, indent=2)


if __name__ == '__main__':
    main()