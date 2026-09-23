# DataForge 1.0 ⚡

DataForge is a modular, automated machine learning pipeline designed for zero-friction data preprocessing and model deployment. It intelligently handles data ingestion, cleaning, dynamic encoding, and automated algorithm selection for both regression and classification tasks.

## 🧠 Core Features

* **Dynamic Preprocessing Routing:** Automatically evaluates column skewness to route numerical features to either IQR or Z-Score outlier clipping.
* **Multivariate Imputation:** Bypasses basic mean/median filling by utilizing `KNNImputer` and `IterativeImputer` to predict and fill missing values based on underlying feature relationships.
* **Smart Categorical Encoding:** Prevents the "dummy variable trap" and memory crashes by automatically routing high-cardinality text columns (>10 unique values) to Frequency Encoding, while using One-Hot Encoding for low-cardinality features.
* **AutoML Model Selection:** Evaluates the target variable to infer the task type, handles class imbalances dynamically via SMOTE, trains a suite of algorithms (LightGBM, Random Forest, Logistic/Linear Regression), and automatically exports the highest-scoring model and its preprocessors as deployment-ready `.pkl` artifacts.

## 📂 Repository Structure

```text
DATAFORGE_1.0/
├── dataforge/                  # Core Python package containing modular pipeline steps
│   ├── data_ingestion.py       # File handling and memory downcasting
│   ├── null_fill.py            # Multivariate imputation logic
│   ├── target_tester.py        # Target inference and Label Encoding
│   └── ...                     
├── run_pipeline.py             # Main orchestrator script
├── requirements.txt            # Dependency list
└── README.md

⚙️ Installation
Clone the repository and install the required dependencies:

git clone [https://github.com/Aaditya-229/DataForge_1.0.git](https://github.com/Aaditya-229/DataForge_1.0.git)
cd DataForge_1.0
pip install -r requirements.txt

🚀 Quickstart
To run the automated pipeline on your own dataset, simply execute the main orchestrator from your terminal:

python run_pipeline.py

The script will trigger a native file dialog for you to select your .csv or .xlsx file, prompt you for the target column, and autonomously execute the pipeline through to model export.



