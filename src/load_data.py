import pandas as pd


def load_vine_data():
    # File paths
    red_wine_path = "data/winequality-red.csv"
    white_wine_path = "data/winequality-white.csv"
    # Loading the datasets with the correct delimiter ( the delimiter is set to ; , which matches the dataset format.
    red_wine = pd.read_csv(red_wine_path, delimiter=';')
    white_wine = pd.read_csv(white_wine_path, delimiter=';')
    return (red_wine, white_wine)