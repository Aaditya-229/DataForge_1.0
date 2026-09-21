import pandas as pd
import numpy as np

def dynamic_outlier_clipping(data, skew_threshold=1.0, z_thresh=3.0, iqr_factor=1.5):
    
    df_clipped = data.copy()
    num_cols = df_clipped.select_dtypes(include=[np.number]).columns
    
    print("Starting dynamic outlier clipping...\n")
    
    for col in num_cols:
        skewness = df_clipped[col].skew()
        
        if abs(skewness) > skew_threshold:
            # High skew -> Route to IQR
            q1 = df_clipped[col].quantile(0.25)
            q3 = df_clipped[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - (iqr_factor * iqr)
            upper = q3 + (iqr_factor * iqr)
            strategy = "IQR"
            
        else:
            # Normal distribution -> Route to Z-Score
            mean = df_clipped[col].mean()
            std = df_clipped[col].std()
            lower = mean - (z_thresh * std)
            upper = mean + (z_thresh * std)
            strategy = "Z-Score"
            
        # Apply the bounds
        df_clipped[col] = df_clipped[col].clip(lower=lower, upper=upper)
        print(f" -> [{strategy}] applied to '{col}' (Skewness: {skewness:.2f})")
        
    print("\n" + "=" * 120)
    return df_clipped