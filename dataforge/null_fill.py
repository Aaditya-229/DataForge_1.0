import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, KNNImputer

def multivariate_imputation(data, strategy='knn', n_neighbors=5, max_iter=10):
    
    df_imputed = data.copy()
    
    num_cols = df_imputed.select_dtypes(include=[np.number]).columns
    
    missing_count = df_imputed[num_cols].isnull().sum().sum()
    if missing_count == 0:
        print("No missing values found in numerical columns. Skipping imputation.")
        return df_imputed

    print(f"Starting multivariate imputation using '{strategy.upper()}' strategy...")
    print(f" -> Found {missing_count} missing values to impute.\n")

    if strategy == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    elif strategy == 'iterative':
   
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    else:
        print(f"❌ Strategy '{strategy}' not recognized. Defaulting to 'knn'.")
        imputer = KNNImputer(n_neighbors=n_neighbors)


    imputed_array = imputer.fit_transform(df_imputed[num_cols])
    df_imputed[num_cols] = pd.DataFrame(imputed_array, columns=num_cols, index=df_imputed.index)
    
    print(" -> Imputation complete.")
    print("\n" + "=" * 55)
    
    return df_imputed