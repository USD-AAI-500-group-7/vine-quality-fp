import pprint

def integrity_report(red_wine, white_wine, skip_data_types=False):
    print("********************************Integrity Report**********************")
    print("\n**********Red Wine:**********\n")
    print("Missing Values:")
    print(red_wine.isnull().sum()) # 1- we check for missing values in both datasets
    print("\nDuplicate Rows:")
    print(red_wine.duplicated().sum()) # 2- We check for the duplicate rows
    if not skip_data_types:
        print("\nData Types:")
        print(red_wine.dtypes) # 3- Check for data types

    print("\n**********White Wine:**********\n")
    print("Missing Values:")
    print(white_wine.isnull().sum())
    print("\nDuplicate Rows:")
    print(white_wine.duplicated().sum())
    if not skip_data_types:
        print("\nData Types:")
        print(white_wine.dtypes)

    print("********************************Integrity Report End**********************")
