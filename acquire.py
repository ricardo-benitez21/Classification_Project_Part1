import pandas as pd
import numpy as np

import env
import os

def check_file_exists(filename, query, url):
    '''
    this function will take a filename, query, and url to check if a file exists
    if it works it will load the dataset from either a sql database or from the
    local .csv file
    '''
    if os.path.exists(filename):
        print('this file exists, reading csv')
        df = pd.read_csv(filename, index_col=0)
    else:
        print('this file doesnt exist, read from sql, and export to csv')
        df = pd.read_sql(query, url)
        df.to_csv(filename)
        
    return df

def get_telco_data():
    '''
    this fucnction will return a dataframe containing the joins below from the telco_churn database
    '''
    url = env.get_db_url('telco_churn')
    query = '''
    select *
    from customers
        join contract_types
            using (contract_type_id)
        join internet_service_types
            using (internet_service_type_id)
        join payment_types
            using (payment_type_id)
    '''
    
    filename = 'telco_churn.csv'

    #call the check_file_exists fuction 
    df = check_file_exists(filename, query, url)
    return df
  