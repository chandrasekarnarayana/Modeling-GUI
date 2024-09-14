from sklearn.preprocessing import StandardScaler

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

