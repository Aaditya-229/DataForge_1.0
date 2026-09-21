# regression_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def reg_model(input_val: pd.DataFrame, target_col: pd.Series):
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

    X_train, X_test, y_train, y_test = train_test_split(df_processed, target_col, test_size=0.2, random_state=42)
    
    print("Applying standardization...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
    }

    best_score = -float('inf')
    best_model = None
    best_name = ""

    print("\nTraining models and evaluating R2 scores...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = round(r2_score(y_test, pred), 3)
        print(f" -> {name}: {score}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print(f"\n✅ DataForge automatically selected '{best_name}' for deployment with an R2 of {best_score}.")
    print("-" * 55)
    
    return best_model, best_name, original_columns, feature_names, scaler