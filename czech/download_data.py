# std library
from io import FileIO
from collections import defaultdict
from pprint import pprint
from sys import stderr
from time import sleep
from datetime import datetime

# download needed
import pandas as pd
import numpy as np
import requests

def load_key() -> str:
    return FileIO('czech_api_token.txt').readall().decode('utf-8')

def download_most_data(base_url: str, query_params_base: dict[str, any], headers: dict[str, str], sleep_const: float):
    url = base_url + 'nakazeni-vyleceni-umrti-testy'
    master_dict = defaultdict(lambda: list())

    page_num = 1
    while True:
        query_params = query_params_base | {'page': page_num}

        resp = requests.get(url, params=query_params, headers=headers)
        if resp.status_code == 200:
            json_resp: list[dict[str, any]] = resp.json()
            if len(json_resp) == 0:
                break
            for item in json_resp:
                if len(item.keys()) != 11:
                    print(f'Invalid object: {item}', file=stderr)
                for key in item:
                    master_dict[key].append(item[key])
            print(f'Loaded page {page_num} of sick, cured, died, tests')
        else:
            break
        
        # sleep to avoid surpassing limit 1000 calls/hour
        sleep(sleep_const)
        page_num += 1
        # if page_num == 2:
        #     break
        
    df = pd.DataFrame(data=master_dict)
    df.to_csv('master.csv', encoding='utf-8', index=False)

def download_vaccine_data(base_url: str, query_params_base: dict[str, any], headers: dict[str, str], sleep_const: float):
    url = base_url + 'ockovani'
    # key order: date -> [first, second]
    master_dict = defaultdict(lambda: defaultdict(lambda: 0))

    page_num = 1
    while True:
        query_params = query_params_base | {'page': page_num}
        resp = requests.get(url, params=query_params, headers=headers)
        if resp.status_code == 200:
            resp_json: list[dict[str, any]] = resp.json()
            if len(resp_json) == 0:
                break

            for item in resp_json:
                current_dict = master_dict[item['datum']]

                current_dict['first'] += item['prvnich_davek']
                current_dict['second'] += item['druhych_davek']
            
            print(f'Loaded page {page_num} of vaccines')
        else:
            break
        
        # sleep to avoid surpassing limit 1000 calls/hour
        sleep(sleep_const)
        page_num += 1
        # if page_num == 2:
        #     break
    
    df_dict = defaultdict(lambda: list())
    for date in master_dict:
        first_doses = master_dict[date]['first']
        second_doses = master_dict[date]['second']

        df_dict['datum'].append(date)
        df_dict['first_vaccine'].append(first_doses)
        df_dict['second_vaccine'].append(second_doses)
    
    del master_dict

    df = pd.DataFrame(data=df_dict)

    df.to_csv('vaccines.csv', index=False)

def download_hospitalization(base_url: str, query_params_base: dict[str, any], headers: dict[str, str], sleep_const: float):
    url = base_url + 'hospitalizace'
    master_dict = defaultdict(lambda: list())

    page_num = 1
    while True:
        query_params = query_params_base | {'page': page_num}
        resp = requests.get(url, params=query_params, headers=headers)
        if resp.status_code == 200:
            resp_json: list[dict[str, any]] = resp.json()
            if len(resp_json) == 0:
                break

            for item in resp_json:
                for key in item:
                    master_dict[key].append(item[key])
            
            print(f'Loaded page {page_num} of hospitalization')
        else:
            break
        
        # sleep to avoid surpassing limit 1000 calls/hour
        #sleep(sleep_const)
        page_num += 1

    df = pd.DataFrame(data=master_dict)

    for col in df.columns:
        if col in ['id', 'datum']:
            continue

        df[col].fillna(0, inplace=True)
        df[col] = df[col].astype(np.int32)

    df.to_csv('hospitalization.csv', index=False)

def main():
    base_url = 'https://onemocneni-aktualne.mzcr.cz/api/v3/'
    headers = {
        'Accept': 'application/json'
    }
    #api_key = load_key()
    query_params_base = {
        'apiToken': load_key()
    }
    requests_per_minute = 16
    sleep_const = 60 / requests_per_minute
    start_dt = datetime.now()

    download_most_data(base_url, query_params_base, headers, sleep_const)
    download_vaccine_data(base_url, query_params_base, headers, sleep_const)
    download_hospitalization(base_url, query_params_base, headers, sleep_const)

    end_dt = datetime.now()

    with open('download_info.txt', 'w') as download_info_file:
        print(f'Started at {start_dt}', file=download_info_file)
        print(f'Ended at {end_dt}', file=download_info_file)

if __name__ == '__main__':
    main()