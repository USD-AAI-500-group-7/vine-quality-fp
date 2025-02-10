from load_data import load_vine_data

red_wine, white_wine = load_vine_data()

# 1- we check for missing values in both datasets
missing_values_red = red_wine.isnull().sum()
missing_values_white = white_wine.isnull().sum()

# 2- We check for the duplicate rows
duplicate_rows_red = red_wine.duplicated().sum()
duplicate_rows_white = white_wine.duplicated().sum()

# 3- Check for data types
data_types_red = red_wine.dtypes
data_types_white = white_wine.dtypes

# Compile findings
data_cleaning_report = {
    "Red Wine": {
        "Missing Values": missing_values_red,
        "Duplicate Rows": duplicate_rows_red,
        "Data Types": data_types_red
    },
    "White Wine": {
        "Missing Values": missing_values_white,
        "Duplicate Rows": duplicate_rows_white,
        "data Types": data_types_white
    }
}

print(data_cleaning_report)