# 💳 Fraud Transaction Detection using Machine Learning

This project is a **machine learning-based web application** that detects fraudulent financial transactions from a CSV file. It provides a simple UI for uploading transaction data and outputs predictions indicating whether each transaction is fraudulent or legitimate.

## 🔍 Overview

- **Tech Stack:** Python, Flask, Scikit-learn, Pandas, NumPy, Bootstrap
- **Model:** Supervised learning model (Random Forest Classifier)
- **UI:** Built using Flask + HTML + CSS (Bootstrap & custom styles)

## Project structure

Fraud_Detection_Project/ <br>
│
├── data/                         # Raw and cleaned datasets <br>
│   ├── FraudDataset/            # Raw datasets <br>
│   └── FraudDataset_cleaned.csv # Preprocessed dataset <br>
│
├── models/                      # Saved ML models <br>
│   └── fraud_detection_model.pkl <br>
│
├── outputs/                     # Model predictions / logs (optional) <br>
│
├── src/                         # Source code for the application <br>
│   ├── __pycache__/             # Python bytecode cache <br>
│   ├── static/                  # Static files (CSS, JS) <br>
│   ├── uploads/                 # Uploaded CSVs for prediction <br>
│   ├── app.py                   # Flask app entry point <br>
│   ├── Load_and_Explore.py      # Data loading & exploration script <br>
│   ├── Prediction.py            # Prediction logic <br>
│   ├── Preprocess.py            # Data preprocessing module <br>
│   ├── run_model_training.py    # Model training pipeline <br>
│   ├── run_pre_processing.py    # Dataset pre-cleaning script <br>
│   ├── SaveModel.py             # Model saving utilities <br>
│   └── requirements.txt         # Python dependencies <br>
│
├── templates/                   # HTML templates for Flask <br>
│   └── index.html <br>
│
├── readme.md                    # Project overview and instructions <br>
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
📧 Email: rudraksh.sachdeva.19work@gmail.com  
🔗 LinkedIn: [linkedin.com/in/rudraksh-sachdeva](https://www.linkedin.com/in/rudraksh-sachdeva)  
🐱 GitHub: [github.com/rudraksh-sachdeva](https://github.com/rudraksh-sachdev)
