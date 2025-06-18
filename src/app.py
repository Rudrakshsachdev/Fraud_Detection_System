# File for flask related code


from flask import Flask, render_template, request # importing Flask and other necessary modules
import pandas as pd # importing pandas for data manipulation
import os # importing os for file handling
import joblib # importing joblib for loading the model
from Preprocess import PreProcessDataset # Importing the PreProcessDataset function from Preprocess module

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates')) # Creating a Flask application instance and specifying the template folder
UPLOAD_FOLDER = 'uploads' # Creating a folder to store uploaded files
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Creating the folder if it doesn't exist

model = joblib.load('C:\\Fraud_detection_project\\models\\fraud_detection_model.pkl') # Loading the pre-trained model

@app.route('/') # Defining the route for the home page

def HomePage():
    return render_template('index.html') # Rendering the hpme page template

@app.route('/predict', methods=['POST']) # Defining the route for the prediction page

def PredictFraud():
    file = request.files['file'] # Getting the uploaded files from the request
    FilePath = os.path.join(UPLOAD_FOLDER, file.filename) # Constructing the file path 
    file.save(FilePath) # Saving the uploaded file to the specified path

    df = pd.read_csv(FilePath) # Reading the uploaded CSV file into a dataframe
    df = PreProcessDataset(df) # Preprocessing the dataset

    if df is None:
        return "Error in preprocessing the dataset. Please check the file format and content."
    
    predictions = model.predict(df.drop('TX_FRAUD', axis=1, errors='ignore')) # Making the predictions using the loaded model
    df['PREDICTION'] = predictions # Adding the predictions to the dataframe

    result = df[['TX_AMOUNT', 'TX_TIME_SECONDS', 'PREDICTION']].head(10).to_html(classes='table table-bordered', index=False) # Selecting the relevant columns for the result 
    return render_template('index.html', tables=[result], titles='Prediction Results') # Rendering the result in the home page template

if __name__ == '__main__':
    app.run(debug=True) # Running the Flask application in debug mode
    

