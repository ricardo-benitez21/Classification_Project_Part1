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
    One-hot encodes categorical columns in the given DataFrames (train, validate, and test).

    Parameters:
    - train (pd.DataFrame): The training dataset.
    - validate (pd.DataFrame): The validation dataset.
    - test (pd.DataFrame): The test dataset.

    Returns:
    List of Encoded DataFrames:
    - train_encoded (pd.DataFrame): Encoded training dataset.
    - validate_encoded (pd.DataFrame): Encoded validation dataset.
    - test_encoded (pd.DataFrame): Encoded test dataset.

    This function performs one-hot encoding on categorical columns in the provided DataFrames,
    excluding 'customer_id' and 'total_charges'. Dummy variables are created for categorical
    columns using pd.get_dummies, and the original columns are dropped from the DataFrames.

    Example:
    train_encoded, validate_encoded, test_encoded = telco_encoded(train_df, validate_df, test_df)
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