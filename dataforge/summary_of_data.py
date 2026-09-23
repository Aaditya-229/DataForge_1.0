#Summary of data
# Gives a description about data..!
import pandas as pd

def summary_of_data(data):
    columns = data.columns.to_list()
    shape = data.shape
    missing_values = data.isnull().sum()
    duplicates = data.duplicated().sum()

    print(f"The dataset has {shape[0]} rows and {shape[1]} columns.\n")
    print(f"The column names are : {columns}\n")
    print(f"The missing values in each attributes are shown below...!\n\n{missing_values}\n")
    print(F"The dataset has {duplicates} duplicate values.\n")
    print(F"Other things you may want a look into : \n\n{data.describe()}\n")
    print("\n" + "=" * 120)
