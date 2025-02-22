# Creating a function to remove outliers using IQR (without over removal)
def remove_outliers_iqr(data, columns):
    # Calculate Q1, Q3, and IQR for each column
    Q1 = data[columns].quantile(0.25)
    Q3 = data[columns].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile range

    # Define bounds for all columns
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Apply filtering only once (removes fewer rows)
    filtered_data = data[~((data[columns] < lower_bound) | (data[columns] > upper_bound)).any(axis=1)]

    return filtered_data

def remove_outliers(red_wine, white_wine):
    # List of columns to apply outlier removal (only high variance ones)
    outlier_columns = ["total sulfur dioxide", "free sulfur dioxide", "residual sugar"]

    # Applying outlier removal only on selected features
    red_wine_cleaned = remove_outliers_iqr(red_wine, outlier_columns)
    white_wine_cleaned = remove_outliers_iqr(white_wine, outlier_columns)

    # Printing dataset size after removing outliers
    print(f"\nRed wine dataset size after outlier removal: {red_wine_cleaned.shape}")
    print(f"White wine dataset size after outlier removal: {white_wine_cleaned.shape}\n")

    return red_wine_cleaned, white_wine_cleaned