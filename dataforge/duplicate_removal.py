#DUPLICATE REMOVAL..!
import pandas as pd

def remove_duplicates(df):
    initial_count = len(df)
    df = df.drop_duplicates(keep = "first").reset_index(drop=True)
    final_count = len(df)
    print(f"DataForge Removed {initial_count - final_count} duplicate rows.\n")
    print("\n" + "=" * 120)
    return df
