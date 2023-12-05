def prep_telco(df):
    '''
    Drops columns, replace empty space values with 0.0 so it can have a value,
    and changing the values so 'Yes' for rows where 'senior_citizen' is equal to 1,
    and 'No' for rows where 'senior_citizen' is not equal to 1.
    '''
    df = df.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id'])
    df.total_charges = df.total_charges.str.replace(' ', '0.0')
    df.senior_citizen = np.where(df['senior_citizen'] == 1, 'Yes', 'No')
    
    return df

def splitting_data(df, col, seed=123):
    '''
    Splits a DataFrame into training, validation, and test sets using a two-step process.
    '''

    #first split
    train, validate_test = train_test_split(df,
                     train_size=0.6,
                     random_state=seed,
                     stratify=df[col]
                    )
    
    #second split
    validate, test = train_test_split(validate_test,
                                     train_size=0.5,
                                      random_state=seed,
                                      stratify=validate_test[col]
                        
                                     )
    return train, validate, test