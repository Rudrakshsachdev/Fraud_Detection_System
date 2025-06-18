import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from Preprocess import PreProcessDataset # Importing the PreProcessDataset function from Preprocess module


#Setting display and visualization
pd.set_option('display.max_columns', None) # show all the columns in the dataset
sns.set(style='whitegrid') # Setting the style of seaborn plots

#function to load the dataset
def LoadDataset(folder_path):
    try:
        all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pkl')]) #Getting all the pickel files in the folder
        all_dfs = [] #List to store all the dataframes

        for file in all_files: #Iterating through each file
            full_path = os.path.join(folder_path, file) #Constructing the full path of the file
            df = pd.read_pickle(full_path) #Loading a pickel file into the dataframe
            all_dfs.append(df) #Appending the dataframe to the list


        combined_df = pd.concat(all_dfs, ignore_index=True) #Combining all dataframes into a single dataframe
        print("✅ Dataset loaded successfully.")
        return combined_df #Returning the combined dataframe

    except Exception as e: # Handling any exception that occurs during the loading of the dataset
        print("❌ Error Loading dataset:", e)
        return None

#Function to explore the dataset
def ExploreDataset(df):
    if df is not None:
        print("DataSet Information: ")
        df.info()  # Displaying the information of the dataset

        print("\nDataSet Description: ")
        print(df.describe()) # Displaying the statistical description of the dataset

        print("\nDataSet Head: ")
        print(df.head()) # Displaying the first 5 rows of the dataset

        print("\nDataSet Columns: ")
        print(df.columns.tolist()) # Displaying the list of columns in the dataset

        print(f"\nDataSet Shape: {df.shape}") # Displaying the shape of the dataset

        print("\nMissing Values: ")
        print(df.isnull().sum()) # Displaying the count of missing values in each column

        print("\nFraud Label distribution: ")
        print(df['TX_FRAUD'].value_counts()) # Displaying the count of fraud and non-fraud transactions

        #Now, Visualizing the distribution of the fraud label
        plt.figure(figsize=(8,6)) 
        sns.countplot(data=df, x='TX_FRAUD', palette='Set2') #  Counting plot for fraud labels
        plt.title("Fraud vs Legitimate Transactions") #title of the plot
        plt.xlabel("TX_Fraud (0: Legitimate, 1: Fraud)") #x-axis label
        plt.ylabel("Count") #y-axis label
        plt.tight_layout() #Auto-adjusting the layout for better visibility
        plt.show() #show the plot
    else:
        print("DataFrame is None. Please Load the dataset first.")


if __name__ == '__main__': # Main function to execute the code
    dataSetPath = '../data/FraudDataset' #path to the dataset
    df = LoadDataset(dataSetPath) #loading the dataset
    if df is not None: #Checking if the dataset is loaded successfully or not
        ExploreDataset(df) #Exploring the dataset
        preprocessed_df = PreProcessDataset(df)





