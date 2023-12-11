from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def prep_telco(df):
    '''
    Drops unnecessary columns
    Replace empty space values with 0.0 so it can have a value,
    Filled in null values in internet_service_type
    '''
    df = df.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id'])
    df.total_charges = df.total_charges.replace(' ', '0.0').astype('float')
    df['internet_service_type'] = df['internet_service_type'].fillna('None')
    df = df.set_index('customer_id')
    
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


def preprocess_telco(train, validate, test):
    '''
    preprocess_telco will take in three pandas dataframes
    of our telco data, expected as cleaned versions of this 
    telco data set (see documentation on acquire.py and prepare.py)
    
    output:
    encoded, ML-ready versions of our clean data, with 
    columns sex and embark_town encoded in the one-hot fashion
    return: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    '''
    # with a looping structure:
    # go through the three dfs
    for df in [train, validate, test]:
        df['total_charges'] = df['total_charges'].astype(float)
    # initialize an empty list to see what needs to be encoded:
    encoding_vars = []
    # loop through the columns to fill encoded_vars with appropriate
    # datatype field names
    for col in train.columns:
        if train[col].dtype == 'O':
            encoding_vars.append(col)
    
    # initialize an empty list to hold our encoded dataframes:
    encoded_dfs = []
    for df in [train, validate, test]:
        df_encoded_cats = pd.get_dummies(
            df[encoding_vars],
              drop_first=True).astype(int)
        encoded_dfs.append(pd.concat(
            [df,
            df_encoded_cats],
            axis=1).drop(columns=encoding_vars))
    return encoded_dfs

def compute_class_metrics(y_train, y_pred):
    '''
   Compute various classification metrics based on predicted and true class labels.

    Parameters:
    - y_train (array-like): True class labels for the training set.
    - y_pred (array-like): Predicted class labels for the training set.

    Returns:
    None: Prints the following classification metrics:
        - Accuracy
        - True Positive Rate (Sensitivity/Recall/Power)
        - False Positive Rate (False Alarm Ratio/Fall-out)
        - True Negative Rate (Specificity/Selectivity)
        - False Negative Rate (Miss Rate)
        - Precision (Positive Predictive Value)
        - F1 Score
        - Support for positive and negative classes.
    '''
    counts = pd.crosstab(y_train, y_pred)
    TP = counts.iloc[1,1]
    TN = counts.iloc[0,0]
    FP = counts.iloc[0,1]
    FN = counts.iloc[1,0]
    
    
    all_ = (TP + TN + FP + FN)

    accuracy = (TP + TN) / all_

    TPR = recall = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TNR = TN / (FP + TN)
    FNR = FN / (FN + TP)

    precision =  TP / (TP + FP)
    f1 =  2 * ((precision * recall) / ( precision + recall))

    support_pos = TP + FN
    support_neg = FP + TN
    
    print(f"Accuracy: {accuracy}\n")
    print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
    print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
    print(f"False Negative Rate/Miss Rate: {FNR}\n")
    print(f"Precision/PPV: {precision}")
    print(f"F1 Score: {f1}\n")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")