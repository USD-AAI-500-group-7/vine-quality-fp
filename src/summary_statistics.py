from load_data import load_wine_data_from_local
from prettytable import PrettyTable

def print_table_summary(wine_data):
    # Create a PrettyTable object
    table = PrettyTable(align='l')
    # Define column names
    table.field_names = ["Feature", "Summary"]
    for column in wine_data.columns:
        feature_min = wine_data[column].min()
        feature_max = wine_data[column].max()
        mean = wine_data[column].mean()
        std = wine_data[column].std()
        table.add_row([column, f"mean({mean:.2f});std({std:.2f});min({feature_min:.2f});max({feature_max:.2f})"])
    print(table)

def summary_statistics(red_wine, white_wine):
    # Summary Statistics for Red and White Wine datasets
    print("Red Wine Summary:")
    print(round(red_wine.describe(), 2))
    print_table_summary(red_wine)

    print("\nWhite Wine Summary:")
    print(round(white_wine.describe(), 2))
    print_table_summary(white_wine)



red_wine, white_wine = load_wine_data_from_local()
summary_statistics(red_wine, white_wine)
