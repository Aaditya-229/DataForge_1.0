import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visual_Module(data):
    data_vis = data.select_dtypes(include=[np.number]).copy()
    print("A Histogram of the given dataset is given below..\n")
    data_vis.hist(bins=10, figsize=(10, 10))
    plt.show()
    print("\n" + "=" * 120)
    
    print("A Heatmap of the given dataset is given below..\n")
    corr = data.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(8,7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    
    plt.title("Correlation Heatmap")
    plt.show()
    print("\n" + "=" * 120)
