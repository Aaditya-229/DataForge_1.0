#column_remover
import pandas as pd

def Remove_col(data):
    try:
        user_input = input("Enter the column names that need to be deleted (example: A,B,C) or 'No':\n")
        
        if user_input.strip().lower() == 'no':
            print("Lets go to the next step..!")
            return data
        else:
            cols = [col.strip() for col in user_input.split(',')]
            data = data.drop(columns=cols)
            print("The mentioned columns got dealt with..!")
            return data 
            
    except KeyError as e:
        print(f"Error: Could not find column {e}. Returning original dataset.")
        return data
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return data