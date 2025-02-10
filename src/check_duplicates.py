def check_duplicates(red_wine, white_wine):
    # Check duplicate rows
    # First we check if the duplicate rows in both datasets have the same values across all columns or no, if it was the same
    # then we can remove the duplicates rows.
    red_wine_duplicates = red_wine.duplicated()
    white_wine_duplicates = white_wine.duplicated()
    print(f"Number of duplicates in red wine: {red_wine_duplicates.sum()}")
    print(red_wine[red_wine.duplicated()].head())

    print(f"Number of duplicates in white wine: {white_wine_duplicates.sum()}")
    print(white_wine[white_wine.duplicated()].head())
