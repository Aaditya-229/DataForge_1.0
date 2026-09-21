import pandas as pd

def input_output_separator(df: pd.DataFrame, target_column: str):

    if target_column not in df.columns:
        raise KeyError(f"❌ Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")
        

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"✅ Successfully separated features (X) and target (y: '{target_column}')")
    print("-" * 55)
    
    return X, y