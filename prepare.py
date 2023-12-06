from sklearn.model_selection import train_test_split
import numpy as np

def prep_telco(df):
    '''
    Drops columns, replace empty space values with 0.0 so it can have a value,
    Filled in null values in internet_service_type
    Converted numerical values (1 and 0) in the 'senior_citizen' column into corresponding string labels ('Yes' and 'No').
    '''
    df = df.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id'])
    df.total_charges = df.total_charges.str.replace(' ', '0.0')
    df['internet_service_type'] = df['internet_service_type'].fillna(value='No Internet Service')
    df.senior_citizen = np.where(df['senior_citizen'] == 1, 'Yes', 'No')
    return df

def splitting_data(df, col):
    '''
    Splits a DataFrame into training, validation, and test sets using a two-step process.
    '''

    #first split
    train, validate_test = train_test_split(df,
                     train_size=0.6,
                     random_state=123,
                     stratify=df[col]
                    )
    
    #second split
    validate, test = train_test_split(validate_test,
                                     train_size=0.5,
                                      random_state=123,
                                      stratify=validate_test[col]
                        
                                     )
    return train, validate, test


    def telco_encoded(train, validate, test):
        """
    One-hot (For each unique category in a categorical variable, a new binary column (dummy variable) is created.) encodes categorical columns in the input DataFrames (train, validate, and test),
    excluding specific columns ('customer_id' and 'total_charges') from the encoding process.

    Parameters:
    - train (pd.DataFrame): Training dataset.
    - validate (pd.DataFrame): Validation dataset.
    - test (pd.DataFrame): Test dataset.

    Returns:
    List of pd.DataFrames:
    - train_encoded: Training dataset with one-hot encoded categorical columns.
    - validate_encoded: Validation dataset with one-hot encoded categorical columns.
    - test_encoded: Test dataset with one-hot encoded categorical columns.
    """
         encoded_dfs = []
    for df in [train, validate, test]:
        df_encoded = df.copy()
        for col in df.columns:
            if col == 'customer_id':
                continue
            if col == 'total_charges':
                continue
            elif df[col].dtype == 'O':  
                df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
                df_encoded = df_encoded.join(df_dummies).drop(columns=[col])
        encoded_dfs.append(df_encoded)
    return encoded_dfs