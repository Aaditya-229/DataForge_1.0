#Main_module
import os
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

from dataforge.data_ingestion import DataIngestion
from dataforge.summary_of_data import summary_of_data
from dataforge.column_remover import remove_columns
from dataforge.null_fill import multivariate_imputation
from dataforge.outlier_clipper import dynamic_outlier_clipping
from dataforge.duplicate_removal import remove_duplicates
from dataforge.visual import visual_Module
from dataforge.target_parser import input_output_separator
from dataforge.target_tester import target_processor
from dataforge.regression import reg_model
from dataforge.classification_scaling  import Cls_model_prep
from dataforge.classification_modelling import cls_model_train
from dataforge.export import export_dataforge_artifacts




DataIngestion.print_banner()
data = DataIngestion.data_reading(input("Enter the dataset path or leave blank to select via GUI:\n"))
new_data = DataIngestion.memory_downcasting(data)
summary_of_data(new_data)

cols_input = input("\nEnter columns to drop (comma-separated), or press Enter to skip: ").strip()
cols_to_drop = [c.strip() for c in cols_input.split(",")] if cols_input else None

col_removed_data = remove_columns(new_data, cols_to_drop)
multivariate_imputed_data = multivariate_imputation(col_removed_data, strategy='knn', n_neighbors=5, max_iter=10)
clipped_data = dynamic_outlier_clipping(multivariate_imputed_data)
duplicate_removed_data = remove_duplicates(clipped_data)
visual_Module(duplicate_removed_data)
input_data, target = input_output_separator(duplicate_removed_data, target_column=input("Enter the target column name:\n"))
new_target, task_type, label_encoder = target_processor(target, task_type="auto")

save_artifacts = True if input("Do you want to save the model artifacts? (y/n):\n").strip().lower() == 'y' else False 

if task_type == "r":

    Best_model, Model_name, original_columns, feature_names, Scaler = reg_model(input_data, new_target)
    
    if save_artifacts:
        export_dataforge_artifacts(
            best_model=Best_model, 
            model_name=Model_name, 
            feature_names=feature_names, 
            original_columns=original_columns, 
            scaler=Scaler 
        )
        
elif task_type == "c":
    X_train, X_test, Y_train, Y_test, Scaler, feature_names, original_columns = Cls_model_prep(input_data, new_target)
    Best_model, Model_name = cls_model_train(X_train, X_test, Y_train, Y_test)

    if save_artifacts:
        export_dataforge_artifacts(
            best_model=Best_model, 
            model_name=Model_name, 
            feature_names=feature_names, 
            original_columns=original_columns, 
            scaler=Scaler, 
            label_encoder=label_encoder 
        )

#if save_artifacts:
#   predict(task_type)