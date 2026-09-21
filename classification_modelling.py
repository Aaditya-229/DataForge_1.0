import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

def cls_model_train(X_train, X_test, Y_train, Y_test):
    

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, Y_train)
    log_pred = log_model.predict(X_test)

    rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_model.fit(X_train, Y_train)
    rf_pred = rf_model.predict(X_test)

    lgbm_model = LGBMClassifier(n_estimators=500, random_state=42, verbose=-1)
    lgbm_model.fit(X_train, Y_train)
    lgbm_pred = lgbm_model.predict(X_test)

    
    models = {
        "Logistic Regression": (log_model, log_pred),
        "Random Forest": (rf_model, rf_pred),
        "LightGBM": (lgbm_model, lgbm_pred)
    }

    best_score = -float('inf')
    best_model = None
    best_name = ""

    print("DataForge has trained three classification models. Here are the results:\n")

    for name, (model, pred) in models.items():
        acc = accuracy_score(Y_test, pred)
        
        print("=" * 55)
        print(f"  {name} | Accuracy: {round(acc * 100, 2)}%")
        print("=" * 55)
        print(classification_report(Y_test, pred))
        
       
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    print(f"\n✅ DataForge automatically selected '{best_name}' for deployment with an accuracy of {round(best_score * 100, 2)}%.")
    print("-" * 55)
    
    return best_model, best_name