# DataForge

DataForge is a modular, production-grade Python library designed to automate the heavy lifting of data science workflows. It transforms experimental data preprocessing and modeling scripts into a robust, object-oriented pipeline. 

By standardizing how datasets are cleaned, scaled, and evaluated, DataForge eliminates boilerplate code and prevents common pitfalls like data leakage during model training.

## Current Focus: DataForge 1.0 (Core ML Engine)
The current iteration focuses on hardening the underlying machine learning logic into a reliable production package. 
* **Automated Preprocessing:** Object-oriented modules for dynamic missing data imputation, outlier clipping, and feature scaling.
* **Pipeline Integration:** Custom components built to integrate seamlessly with scikit-learn's `Pipeline` API.
* **Model Selection:** Automated routing and hyperparameter optimization for classification models (Logistic Regression, SVM, Random Forest).

## Roadmap: DataForge 2.0 (Agentic Orchestration)
Future releases will evolve the core engine into an intelligent state machine.
* **LangChain & LangGraph:** Introducing reasoning agents to dynamically analyze datasets and orchestrate the preprocessing nodes based on contextual context.
* **Django Backend:** Wrapping the pipeline into a REST API for web-based execution and natural language querying.
