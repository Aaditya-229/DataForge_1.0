import pandas as pd
import numpy as np
import os

class DataIngestion:

    @staticmethod
    def print_banner():
        print(r"""
     _       __     __                             __      
    | |     / /__  / /_________  ____ ___  ___    / /_____ 
    | | /| / / _ \/ / ___/ __ \/ __ `__ \/ _ \  / __/ __ \
    | |/ |/ /  __/ / /__/ /_/ / / / / / /  __/ / /_/ /_/ /
    |__/|__/\___/_/\___/\____/_/ /_/ /_/\___/  \__/\____/ 
                                                          
        ____        __        ______                       
       / __ \____ _/ /_____ _/ ____/___  _________ ____    
      / / / / __ `/ __/ __ `/ /_  / __ \/ ___/ __ `/ _ \   
     / /_/ / /_/ / /_/ /_/ / __/ / /_/ / /  / /_/ /  __/   
    /_____/\__,_/\__/\__,_/_/    \____/_/   \__, /\___/    
                                           /____/          
                     ___     ____ 
                   <  /    / __ \
                   / /    / / / /
                  / /_ _ / /_/ / 
                 /_/(_|_)\____/  

    ======================================================
             Automated Machine Learning Pipeline
    ======================================================
    """)


    @staticmethod
    def file_selection():
        """Optional GUI fallback for local file selection."""
        
        import tkinter as tk
        from tkinter import filedialog
        
        
        initial_dir = os.path.expanduser('~')
        
        root = tk.Tk() 
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select a dataset", 
            initialdir=initial_dir, 
            filetypes=(("CSV files", "*.csv"), ("Excel files", "*.xlsx"))
        )
        root.destroy()
        return file_path

    @staticmethod
    def data_reading(filepath=None): 
        """
        Reads data from a provided filepath. 
        If no filepath is provided, it safely falls back to a GUI file dialog.
        """
        if not filepath:
            print("No filepath provided. Opening file dialog...")
            filepath = DataIngestion.file_selection()
            
        if not filepath:
            raise ValueError("❌ No file selected or provided. Aborting ingestion.")

        try:
            if filepath.endswith(".csv"):
                data = pd.read_csv(filepath)
            elif filepath.endswith((".xls", ".xlsx")):
                data = pd.read_excel(filepath)
            else:
                raise ValueError(f"❌ Unsupported file format for {filepath}. Must be .csv or .xlsx")
            
            print(f"\n✅ Dataset '{os.path.basename(filepath)}' successfully loaded.")
            print(f"Preview:\n{data.head()}\n" + "=" * 55)
            return data
            
        except Exception as e:
            raise RuntimeError(f"❌ Error reading file: {e}")
    
    @staticmethod
    def memory_downcasting(data):
        Mem = data.copy()
        for col in Mem.select_dtypes(include=['int64']).columns:
            Mem[col] = pd.to_numeric(Mem[col], downcast='integer')
        for col in Mem.select_dtypes(include=['float64']).columns:
            Mem[col] = pd.to_numeric(Mem[col], downcast='float')
        print("✅ Memory downcasting completed.\n" + "-" * 55)
        return Mem