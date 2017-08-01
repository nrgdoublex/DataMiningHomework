import pandas as pd
import numpy as np
from six import string_types

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def float_or_str(x):
    if isfloat(x):
        return (x)
    else:
        return (-1)


def percent_to_float(x):
    if isfloat(x):
        return (x/100)
    else:
        return float(x.strip('%'))/100

def add_noise(x):
    if not isinstance(x, string_types):
        return (x + np.random.normal(loc=0.0, scale=1e-3))
    else:
        return (x)


def data_processing(df):

    df['Interest Rate Percentage'] = [percent_to_float(i) for i in df['Interest Rate Percentage']]

    df['Debt-To-Income Ratio'] = [percent_to_float(i) for i in df['Debt-To-Income Ratio Percentage']]

    features_to_keep = ['Amount Requested','Interest Rate Percentage','Loan Purpose','Loan Length in Months',
                        'Monthly PAYMENT','Total Amount Funded','FICO Range','Debt-To-Income Ratio Percentage']

    # convert interger values to float (helps avoiding optimization implementation issues)
    for feature in features_to_keep:
        if feature not in ['FICO Range','Loan Purpose']:
            df[feature] = [float(i) for i in df[feature]]

    # Scale values
    df['Total Amount Funded'] /= max(df['Total Amount Funded'])
    df['Amount Requested'] /= max(df['Amount Requested'])
    df['Loan Length in Months'] /= max(df['Loan Length in Months'])
    df['Monthly PAYMENT'] /= max(df['Monthly PAYMENT'])

    # Interaction terms
    df['Total Amount Funded * Requested'] = df['Total Amount Funded']*df['Amount Requested']
    df['Total Amount Funded * Requested'] /= max(df['Total Amount Funded * Requested'])

    df['Interest Rate Percentage * Monthly PAYMENT'] = df['Interest Rate Percentage']*df['Monthly PAYMENT']
    df['Interest Rate Percentage * Monthly PAYMENT'] /= max(df['Interest Rate Percentage * Monthly PAYMENT'])


    target_var = [float_or_str(i) for i in df['Status']]

    # create a clean data frame for the regression
    data = df[features_to_keep].copy()
    
    data['intercept'] = 1.0

    return (data,target_var)

def add_categorical(train, validation, feature_str):
    # encode categorical features
    encoded = pd.get_dummies(pd.concat([train[feature_str],validation[feature_str]], axis=0))#, dummy_na=True)
    train_rows = train.shape[0]
    train_encoded = encoded.iloc[:train_rows, :]
    validation_encoded = encoded.iloc[train_rows:, :] 

    train_encoded_wnoise = train_encoded.applymap(add_noise)
    validation_encoded_wnoise = validation_encoded.applymap(add_noise)

    train.drop(feature_str,axis=1, inplace=True)
    validation.drop(feature_str,axis=1, inplace=True)

    train = train.join(train_encoded_wnoise.ix[:, :])
    validation = validation.join(validation_encoded_wnoise.ix[:, :])

    return (train,validation)