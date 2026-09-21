import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def Cls_model_prep(input_val: pd.DataFrame, target_col: pd.Series, auto_smote=True):
    df_processed = input_val.copy()
    
    print("Evaluating categorical cardinality...")
    cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        unique_count = df_processed[col].nunique()
        if unique_count >= 10:
            freq_map = df_processed[col].value_counts(normalize=True)
            df_processed[col] = df_processed[col].map(freq_map)
            print(f" -> '{col}' ({unique_count} unique) converted via Frequency Encoding.")
            
    
    df_processed = pd.get_dummies(df_processed, drop_first=True, dtype=float)
    
    original_columns = input_val.columns.tolist()
    feature_names = df_processed.columns.tolist()
    
    X_train, X_test, Y_train, Y_test = train_test_split(df_processed, target_col, test_size=0.2, random_state=42)
    
    
    if Y_train.dtype == 'object' or Y_train.dtype == 'bool' or Y_train.dtype.name == 'category':
        le = LabelEncoder()
        Y_train = pd.Series(le.fit_transform(Y_train), index=Y_train.index)
        Y_test = pd.Series(le.transform(Y_test), index=Y_test.index)
        
    
    class_counts = Y_train.value_counts()
    imbalance_ratio = class_counts.min() / class_counts.max()
    
    if auto_smote and imbalance_ratio < 0.5:
        print(f"\nClass imbalance detected (Ratio: {imbalance_ratio:.2f}). Applying SMOTE...")
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)
        print(" -> SMOTE applied successfully.")
    else:
        print(f"\nClass distribution is balanced enough (Ratio: {imbalance_ratio:.2f}). Skipping SMOTE.")
        
    print("\nApplying standardization...")
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print("✅ DataForge prep complete for classification models.")
    print("-" * 55)
    
    return X_train, X_test, Y_train, Y_test, scale, feature_names, original_columns