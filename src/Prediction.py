import joblib # Importing Joblib for loading the save model
import pandas as pd # Importing pandas for data manipulation

# Function to load the model from the specified path
def LoadModel(model_Path):
    try:
        model = joblib.load(model_Path) #Loading the saved model from the specified path
        print("✅ Model loaded successfully.")
        return model #Returning the loaded model
    except Exception as e: # Handling any exception that occurs during loading the model
        print("❌ Error Loading model:", e)
        return None
    

# Function to make predictions using the loaded model
def MakePrediction(model, data):
    try:
        predictions = model.predict(data) # Making predicitions from the loaded model
        print("✅ Predictions made successfully.")
        return predictions #Returning the predictions
    except Exception as e: # Handling any exception that occurs during making predictions'
        print("❌ Error Making predictions:", e)
        return None

if __name__ == "__main__":
    ModelPath = '../models/fraud_detection_model.pkl'
    model = LoadModel(ModelPath)  # Load the model from the specified path
    if model is not None:
        # Load the dataset for making predictions
        data_path = '../data/FraudDataset_cleaned.pkl' 
        df = pd.read_pickle(data_path)  # Load the preprocessed dataset

        # Make predictions using the loaded model
        predictions = MakePrediction(model, df.drop('TX_FRAUD', axis=1))  # Exclude target variable from features
        if predictions is not None:
            print("Predictions:", predictions)  # Display the predictions

print("Model trained on features:", model.feature_names_in_)

