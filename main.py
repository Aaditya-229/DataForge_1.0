#Main_module
import os
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())

from data_ingestion import DataIngestion
from summary_of_data import summary_of_data
from column_remover import Remove_col
from null_fill import multivariate_imputation
from outlier_clipper import dynamic_outlier_clipping
from duplicate_removal import remove_duplicates
from visual import visual_Module
from target_parser import input_output_separator
from target_tester import target_processor
from regression import reg_model
from classification_scaling  import Cls_model_prep
from classification_modelling import cls_model_train
from export import export_dataforge_artifacts





file_name = DataIngestion.file_selection()
data = DataIngestion.data_reading(file_name)
new_data = DataIngestion.memory_downcasting(data)
summary_of_data(new_data)
col_removed_data = Remove_col(new_data)
multivariate_imputed_data = multivariate_imputation(col_removed_data, strategy='knn', n_neighbors=5, max_iter=10)
clipped_data = dynamic_outlier_clipping(multivariate_imputed_data)
duplicate_removed_data = remove_duplicates(clipped_data)
visual_Module(duplicate_removed_data)
input_data, target = input_output_separator(duplicate_removed_data, target_column=input("Enter the target column name:\n"))
new_target, task_type, label_encoder = target_processor(target, task_type="auto")

save_artifacts = True  

if task_type == "r":
    # Unpack the scaler here as well!
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