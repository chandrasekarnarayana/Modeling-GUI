from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def normalize_data(df, columns):
    """
    Normalize the specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to normalize.

    Returns:
    pd.DataFrame: A DataFrame with normalized columns.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the DataFrame by filling them with a specific strategy.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): The filling strategy ('mean', 'median', 'mode').

    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'mode'.")


def apply_missing_strategy(df, strategy):
    """
    Apply missing value handling to a DataFrame.
    strategy: 'none', 'drop', 'mean'
    """
    if strategy == 'none':
        return df
    if strategy == 'drop':
        return df.dropna()
    if strategy == 'mean':
        numeric_cols = df.select_dtypes(include=np.number).columns
        return df.copy().fillna(df[numeric_cols].mean())
    raise ValueError("Invalid missing value strategy.")


def standardize_features(df, columns):
    """
    Standardize numeric features and return transformed DataFrame plus scaler.
    """
    scaler = StandardScaler()
    df_copy = df.copy()
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy, scaler


def select_numeric_columns(df, columns):
    """
    Return numeric subset of requested columns and list of dropped non-numeric columns.
    """
    selected = df[columns]
    numeric_df = selected.select_dtypes(include=np.number)
    dropped = [col for col in columns if col not in numeric_df.columns]
    return numeric_df, dropped
