from Load_and_Explore import LoadDataset  # importing from your loadandexplore.py
from Preprocess import PreProcessDataset  # importing from your preprocess.py

if __name__ == "__main__":
    # Load the dataset
    data_path = '../data/FraudDataset'  # Adjust if your file has a .pkl extension
    df = LoadDataset(data_path) # Load the dataset from the specified path

    # Preprocess the dataset
    if df is not None:
        df_clean = PreProcessDataset(df) # Preprocess the loaded dataset

        # Show first few rows of the cleaned data
        print(df_clean.head())

        # Optional: Save the preprocessed dataset for later use
        df_clean.to_pickle('../data/FraudDataset_cleaned.pkl')
