import joblib
import os

def export_dataforge_artifacts(best_model, model_name, feature_names, original_columns, scaler=None, label_encoder=None, export_dir="."):
    """
    Silently exports all necessary pipeline artifacts for deployment.
    """
    print(f"\nExporting DataForge artifacts for {model_name}...")
    
    # Save the core model and feature lists
    joblib.dump(best_model, os.path.join(export_dir, "dataforge_model.pkl"))
    joblib.dump(feature_names, os.path.join(export_dir, "dataforge_features.pkl"))
    joblib.dump(original_columns, os.path.join(export_dir, "dataforge_original_cols.pkl"))
    
    # Save the preprocessing objects (Crucial for the predict module)
    if scaler is not None:
        joblib.dump(scaler, os.path.join(export_dir, "dataforge_scaler.pkl"))
        
    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(export_dir, "dataforge_Lbl_edr.pkl"))
        
    print("✅ All artifacts successfully saved to disk. Ready for deployment.")
    print("-" * 55)