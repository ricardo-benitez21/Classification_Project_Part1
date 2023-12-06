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


    def preprocess_telco(train_df, val_df, test_df):
    '''
    Sets the index of each DataFrame to 'customer_id'.
    Converts the 'total_charges' column to the float data type.
    Identifies categorical variables for one-hot encoding.
    One-hot encodes the identified categorical variables with drop_first=True.
    Concatenates the original DataFrame with the one-hot encoded variables.
    Drops the original categorical variables from the DataFrame
    '''
    # using a for loop:
    # go through the three dfs, set the index to customer id
    for df in [train_df, val_df, test_df]:
        df = df.set_index('customer_id')
        df['total_charges'] = df['total_charges'].astype(float)
    # initialize an empty list to see what needs to be encoded:
    encoding_vars = []
    # looping through the columns to fill encoded_vars with appropriate
    # datatype field names
    for col in train_df.columns:
        if train_df[col].dtype == 'O':
            encoding_vars.append(col)
    encoding_vars.remove('customer_id')
    # create an empty list so it can hold the encoded dataframes:
    encoded_dfs = []
    for df in [train_df, val_df, test_df]:
        df_encoded_cats = pd.get_dummies(
            df[encoding_vars],
              drop_first=True).astype(int)
        encoded_dfs.append(pd.concat(
            [df,
            df_encoded_cats],
            axis=1).drop(columns=encoding_vars))
    return encoded_dfs