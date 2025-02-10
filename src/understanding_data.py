from IPython.display import display
from load_data import load_vine_data

def understanding_data(red_wine, white_wine):
    red_wine, white_wine = load_vine_data()
    # Displaying the first 5 rows of the data (both red and white wine) in table format
    display(red_wine.head())  # Red vine table
    display(white_wine.head())  # White vine table
