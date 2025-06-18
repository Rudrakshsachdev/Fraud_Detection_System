# 💳 Fraud Transaction Detection using Machine Learning

This project is a **machine learning-based web application** that detects fraudulent financial transactions from a CSV file. It provides a simple UI for uploading transaction data and outputs predictions indicating whether each transaction is fraudulent or legitimate.

## 🔍 Overview

- **Tech Stack:** Python, Flask, Scikit-learn, Pandas, NumPy, Bootstrap
- **Model:** Supervised learning model (Random Forest Classifier)
- **UI:** Built using Flask + HTML + CSS (Bootstrap & custom styles)

## Project structure

Fraud_Detection_Project/
│
├── data/                         # Raw and cleaned datasets
│   ├── FraudDataset/            # Raw datasets
│   └── FraudDataset_cleaned.csv # Preprocessed dataset
│
├── models/                      # Saved ML models
│   └── fraud_detection_model.pkl
│
├── outputs/                     # Model predictions / logs (optional)
│
├── src/                         # Source code for the application
│   ├── __pycache__/             # Python bytecode cache
│   ├── static/                  # Static files (CSS, JS)
│   ├── uploads/                 # Uploaded CSVs for prediction
│   ├── app.py                   # Flask app entry point
│   ├── Load_and_Explore.py      # Data loading & exploration script
│   ├── Prediction.py            # Prediction logic
│   ├── Preprocess.py            # Data preprocessing module
│   ├── run_model_training.py    # Model training pipeline
│   ├── run_pre_processing.py    # Dataset pre-cleaning script
│   ├── SaveModel.py             # Model saving utilities
│   └── requirements.txt         # Python dependencies
│
├── templates/                   # HTML templates for Flask
│   └── index.html
│
├── readme.md                    # Project overview and instructions
└── .gitignore                   # Files to be ignored by Git (you can add this)



## 🚀 Features

- Upload transaction data in CSV format
- Preprocessing of raw data including:
  - Datetime parsing and feature engineering
  - Label encoding for customer and terminal IDs
- Fraud detection using a trained machine learning model
- Clean and responsive UI for user interaction
- Displays prediction results in tabular format

## 🧠 Machine Learning Details

- **Algorithm:** Random Forest Classifier
- **Training Features:**
  - `TX_AMOUNT`
  - `TX_TIME_SECONDS`
  - `TX_TIME_DAYS`
  - `TX_FRAUD_SCENARIO`
  - `TX_HOUR`
  - `TX_DAY`
  - `TX_WEEKDAY`
  - `CUSTOMER_ID_ENC`
  - `TERMINAL_ID_ENC`

# 💳 Fraud Transaction Detection using Machine Learning

This project is a **machine learning-based web application** that detects fraudulent financial transactions from a CSV file. It provides a simple UI for uploading transaction data and outputs predictions indicating whether each transaction is fraudulent or legitimate.

## 🔍 Overview

- **Tech Stack:** Python, Flask, Scikit-learn, Pandas, NumPy, Bootstrap
- **Model:** Supervised learning model (Random Forest Classifier)
- **UI:** Built using Flask + HTML + CSS (Bootstrap & custom styles)

## 🚀 Features

- Upload transaction data in CSV format
- Preprocessing of raw data including:
  - Datetime parsing and feature engineering
  - Label encoding for customer and terminal IDs
- Fraud detection using a trained machine learning model
- Clean and responsive UI for user interaction
- Displays prediction results in tabular format

## 🧠 Machine Learning Details

- **Algorithm:** Random Forest Classifier
- **Training Features:**
  - `TX_AMOUNT`
  - `TX_TIME_SECONDS`
  - `TX_TIME_DAYS`
  - `TX_FRAUD_SCENARIO`
  - `TX_HOUR`
  - `TX_DAY`
  - `TX_WEEKDAY`
  - `CUSTOMER_ID_ENC`
  - `TERMINAL_ID_ENC`

## 🔄 How It Works

1. User uploads a transaction CSV.
2. Data is preprocessed (feature extraction & encoding).
3. Preprocessed data is fed into the ML model.
4. Model returns prediction (fraud or not).
5. Results are shown on the UI.

## 🚧 Future Improvements

- Add login/authentication
- Store uploaded files and logs
- Visualize fraud probability with charts
- Deploy to Render/Heroku with database integration

## 👨‍💻 Author

**Rudraksh Sachdeva**  
B.Tech CSE Student | Machine Learning Intern | Tech Enthusiast  
📧 Email: rudraksh.sachdeva@example.com  
🔗 LinkedIn: [linkedin.com/in/rudraksh-sachdeva](https://www.linkedin.com/in/rudraksh-sachdeva)  
🐱 GitHub: [github.com/rudraksh-sachdeva](https://github.com/rudraksh-sachdev)
