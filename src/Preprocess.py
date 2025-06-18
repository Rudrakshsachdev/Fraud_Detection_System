import pandas as pd # Importing pandas for data manipulation and analysis
import numpy as np # Importing numpy for numerical operations
from sklearn.preprocessing import LabelEncoder # Importing LabelEncode for encoding categorical variables

# Function to preprocess the dataset
def PreProcessDataset(df):
    try:
        print('Started Preprocessing the dataset...')

        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'], errors='coerce') # Convert 'TX_DATETIME' to datetime format, coercing errors to NaT
        df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
        df['TX_DAY'] = df['TX_DATETIME'].dt.day
        df['TX_WEEKDAY'] = df['TX_DATETIME'].dt.weekday

        le_Customer = LabelEncoder()
        le_Terminal = LabelEncoder()

        df['CUSTOMER_ID_ENC'] = le_Customer.fit_transform(df['CUSTOMER_ID'])
        df['TERMINAL_ID_ENC'] = le_Terminal.fit_transform(df['TERMINAL_ID'])

        required_cols = [
            'TX_AMOUNT',
            'TX_TIME_SECONDS',
            'TX_TIME_DAYS',
            'TX_FRAUD_SCENARIO',
            'TX_HOUR',
            'TX_DAY',
            'TX_WEEKDAY',
            'CUSTOMER_ID_ENC',
            'TERMINAL_ID_ENC'
        ]

        # Add missing columns with default value
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[required_cols]

        print('✅ Preprocessing completed.')
        return df

    except Exception as e:
        print("❌ Error in Preprocessing dataset:", e)
        return None

    

# Example usage:
# if __name__ == "__main__":
#     df = pd.read_csv('path_to_your_dataset.csv') # Load your dataset here
#     preprocessed_df = PreProcessDataset(df) # Call the preprocessing function
#     if preprocessed_df is not None:
#         print(preprocessed_df.head()) # Display the first few rows of the preprocessed dataframe

