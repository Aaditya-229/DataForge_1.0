import pandas as pd
from dataforge.column_remover import remove_columns

def test_remove_columns_success():
    """Test that valid columns are successfully dropped."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    result = remove_columns(df, cols_to_drop=["B"])
    assert list(result.columns) == ["A", "C"]

def test_remove_columns_skip_missing():
    """Test that the function survives when asked to drop a non-existent column."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = remove_columns(df, cols_to_drop=["C", "D"])
    assert list(result.columns) == ["A", "B"]