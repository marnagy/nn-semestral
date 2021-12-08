import pandas as pd
import numpy as np

def main():
    master_df = pd.read_csv('master.csv')
    vaccines_df = pd.read_csv('vaccines.csv')
    hosp_df = pd.read_csv('hospitalization.csv')

    vacc_hosp_df = pd.merge(vaccines_df, hosp_df, how='outer', on=['datum'])

    #vacc_hosp_df.to_csv('vacc_hosp.csv', index=False)
        
    result_df = pd.merge(master_df, vacc_hosp_df, how='outer', on=['datum'])

    for col in hosp_df.columns:
        if col in ['datum', 'id']:
            continue

        result_df[col].fillna(0, inplace=True)
        result_df[col] = result_df[col].astype(np.int32)
    
    for col in master_df.columns:
        if col == 'datum':
            continue

        result_df[col].fillna(0, inplace=True)
        result_df[col] = result_df[col].astype(np.int32)
    
    for col in vaccines_df.columns:
        if col == 'datum':
            continue

        result_df[col].fillna(0, inplace=True)
        result_df[col] = result_df[col].astype(np.int32)
    
    result_df['datum'] = result_df['datum'].astype(np.datetime64)

    result_df.to_csv('combined.csv', index=False)

if __name__ == '__main__':
    main()