def remove_duplicates(red_wine, white_wine):
    # Our dataset before removing the duplicate rows
    print()
    print("\n---------- Dataset before removing duplicate rows ----------")
    print(f"Red wine dataset size before removing duplicates: {red_wine.shape}")
    print(f"White wine dataset size before removing duplicates: {white_wine.shape}\n")

    # Removing the duplicate rows
    red_wine_cleaned = red_wine.drop_duplicates()
    white_wine_cleaned = white_wine.drop_duplicates()

    # Confirming the changes
    print()
    print("---------- Dataset after removing duplicate rows ----------")
    print(f"Red wine dataset size after removing duplicates: {red_wine_cleaned.shape}")
    print(f"White wine dataset size after removing duplicates: {white_wine_cleaned.shape}\n")
    return red_wine_cleaned, white_wine_cleaned
