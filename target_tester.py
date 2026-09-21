# target_processor.py
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder

def target_processor(target: pd.Series, task_type="auto"):
    
    target_dtype = target.dtype
    unique_num = target.nunique()
    unique_ratio = unique_num / len(target)

    is_numeric = np.issubdtype(target_dtype, np.number)
    is_categorical = target_dtype == "object" or target_dtype.name == "category" or target_dtype == "bool"


    if task_type == "auto":
        if is_categorical or (is_numeric and unique_num <= 20 and unique_ratio < 0.05):
            task_type = "c"
            print("🧠 Auto-detected task: Classification")
        else:
            task_type = "r"
            print("🧠 Auto-detected task: Regression")
            

    else:
        task_type = task_type.lower()
        if task_type == "r" and not (is_numeric and unique_num > 10 and unique_ratio > 0.1):
            warnings.warn("Target has low cardinality or is categorical. Classification may be more suitable.", UserWarning)
        elif task_type == "c" and not (is_categorical or (is_numeric and unique_num <= 20 and unique_ratio < 0.05)):
            warnings.warn("Target has high cardinality. Regression may be more suitable.", UserWarning)

 
    le = None
    if task_type == "c" and is_categorical:
        le = LabelEncoder()
        # Transform and preserve the original pandas index
        target = pd.Series(le.fit_transform(target), index=target.index)
        print(f"✅ Target column encoded. Classes found: {list(le.classes_)}")
        
    print("-" * 55)
    

    return target, task_type, le