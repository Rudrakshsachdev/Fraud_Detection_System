import joblib # Importing joblib for saving and loading models
from Load_and_Explore import LoadDataset # Importing the LoadDataset function from Load_and_Explore module
from Preprocess import PreProcessDataset # Importing the PreProcessDataset function from Preprocess module

from sklearn.ensemble import RandomForestClassifier # Importing RandomForestClassifier from sklearn
from sklearn.model_selection import train_test_split # Importing train_test_split for splitting the dataset

df = LoadDataset('../data/FraudDataset')  # Load the dataset from the specified path
df = PreProcessDataset(df) # Preprocess the loaded dataset

# Step 2: Split features and target
X = df.drop('TX_FRAUD', axis=1)
y = df['TX_FRAUD']

# Step 3: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 4: Save the model
joblib.dump(model, '../models/fraud_detection_model.pkl')
print("✅ Model saved successfully at ../models/fraud_detection_model.pkl")