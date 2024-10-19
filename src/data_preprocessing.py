import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess the DataFrame for clustering.

    This involves:
    - Removing players with overall rating less than 65.
    - Excluding goalkeepers.
    - Handling missing values.
    - Ensuring that all attributes are numeric.

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Preprocessed DataFrame.
        - list: List of attribute column names used for clustering.
    """
    # Filter players based on overall rating and position
    df = df[df['overall'] >= 65]
    df = df[df['best_position'] != 'GK']

    # List of attributes for clustering
    attributes = [
        'crossing', 'finishing', 'heading_accuracy',
        'short_passing', 'volleys', 'dribbling', 'curve',
        'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
        'agility', 'reactions', 'balance', 'shot_power',
        'jumping', 'stamina', 'strength', 'long_shots',
        'aggression', 'interceptions', 'positioning', 'vision',
        'composure', 'marking', 'standing_tackle', 'sliding_tackle'
    ]

    # Drop rows with missing values in these attributes
    df = df.dropna(subset=attributes)

    # Ensure attributes are numeric
    for attribute in attributes:
        df[attribute] = pd.to_numeric(df[attribute], errors='coerce')

    # Drop rows with any remaining NaN values after conversion
    df = df.dropna(subset=attributes)

    return df, attributes
