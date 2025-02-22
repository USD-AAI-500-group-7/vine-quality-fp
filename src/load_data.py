import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_wine_data_from_local():
    # File paths
    red_wine_path = "data/winequality-red.csv"
    white_wine_path = "data/winequality-white.csv"
    # Loading the datasets with the correct delimiter ( the delimiter is set to ; , which matches the dataset format.
    red_wine = pd.read_csv(red_wine_path, delimiter=';')
    white_wine = pd.read_csv(white_wine_path, delimiter=';')
    return (red_wine, white_wine)

def load_wine_data_from_original_source():
    # fetch dataset
    wine_data = fetch_ucirepo(id=186)

    red_wine = wine_data.data.original[wine_data.data.original['color'] == 'red'].reset_index(drop=True)
    white_wine = wine_data.data.original[wine_data.data.original['color'] == 'white'].reset_index(drop=True)

    # Drop unnecessary color column
    red_wine.drop('color', axis=1, inplace=True)
    white_wine.drop('color', axis=1, inplace=True)

    return (red_wine, white_wine)