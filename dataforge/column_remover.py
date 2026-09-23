#column_remover
import pandas as pd

def remove_columns(data: pd.DataFrame, cols_to_drop: list = None) -> pd.DataFrame:
 
    if not cols_to_drop:
        return data
        
    try:
        valid_cols = [col for col in cols_to_drop if col in data.columns]
        return data.drop(columns=valid_cols)
    except Exception as e:
        raise RuntimeError(f"❌ Error removing columns: {e}")