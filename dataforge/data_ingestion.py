import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np

# data IMPORT MODULE..!
class DataIngestion:
    @staticmethod
    def file_selection():
        root = tk.Tk() 
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(title = "Select a file", initialdir = r'C:\Users\User\Downloads', filetypes = (("Excel files", "*.xlsx"),("CSV files", "*.csv")))
        root.destroy()
        return file_path

    @staticmethod
    def data_reading(filepath): 
        
        if filepath:
            data = None
            try:
                if filepath.endswith(".csv"):
                    data = pd.read_csv(filepath)
                elif filepath.endswith(".xlsx"):
                    data = pd.read_excel(filepath)
                else:
                    print("The file needs to be in .csv or .xlsx format\n")
            except PermissionError:
                print("❌ Permission denied while accessing the file.\n")
            except Exception as e:
                print(f"❌ Unexpected error occurred: {e}\n")
            else:
                print(f"\n\nThe dataset has been successfully uploaded, here's a short preview..!\n\n\n{data.head()}")
                print("\n" + "=" * 110)
        else:
            print("No file selected. Aborting ingestion.\n")
            return None
        return data
    
    @staticmethod
    def memory_downcasting(data):
        Mem = data.copy()
        for col in Mem.select_dtypes(include=['int64']).columns:
            Mem[col] = pd.to_numeric(Mem[col], downcast='integer')
        for col in Mem.select_dtypes(include=['float64']).columns:
            Mem[col] = pd.to_numeric(Mem[col], downcast='float')
        print("\n\nMemory downcasting completed.")
        return Mem
