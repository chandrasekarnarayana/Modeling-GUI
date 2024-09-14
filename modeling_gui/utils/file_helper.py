import pandas as pd

def load_csv(file_path):
    """
    Load a CSV file from the specified file path.
    
    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the CSV data.

    Raises:
    FileNotFoundError: If the file is not found.
    ValueError: If the file is not a valid CSV.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except ValueError as e:
        raise ValueError(f"Error reading CSV file: {file_path}") from e

