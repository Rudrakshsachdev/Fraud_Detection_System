import pandas as pd # Importing pandas for data manipulation and analysis
from sklearn.model_selection import train_test_split # Importing train_test_split for splitting the dataset
from sklearn.ensemble import RandomForestClassifier # Importing RandomForestClassifier for model training. This model is suitable for handling highly imbalanced datasets like fraud detection.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #Importing few matrics for evaluating the model performance
from Load_and_Explore import LoadDataset # Importing the LoadDataset function from Load_and_Explore module
from Preprocess import PreProcessDataset # Importing the PreProcessDataset function from Preprocess module

#Load and Preprocess dataset
df = LoadDataset('../data/FraudDataset') #Loading the preprocessed and cleaned dataset
df = PreProcessDataset(df) #Preprocessing the dataset

# STEP 1: Splitting the dataset into training and testing sets and features x and y
X = df.drop('TX_FRAUD', axis=1) # Removing the target variables from datasets and storing the rest of the features in X
Y = df['TX_FRAUD'] # Storing the target variable (0 or 1) in Y

# STEP 2: Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42) # Splitting the dataset into training and testing sets with 80% training and 20% testing
#X_train ensures that the training set contains 80% of the data, while X_test contains 20% of the data.
#Y_train and Y_test are the corresponding target variables for the training and testing sets.
# stratify=Y ensures that the both classes (0 and 1) are balanced in both
# Random_state=42 ensures that the split is reproducible, meaning that every time you run the code, you will get the same split.


# STEP 3: Train the model using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42) # Created a model instance of RandomForestClassifier with 100 trees and a random state for reproducibility
model.fit(X_train, Y_train) # Fitting the model on the training data

# STEP 4: Make predictions on the test set
Y_pred = model.predict(X_test) # Predicting the target variable for the test set

# STEP 5: Evaluate the modelperformance
print("Model Evaluation Metrics: ")
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred)) # Displaying the confusion matrix
print("\nClassification Report: ", classification_report(Y_test, Y_pred)) # Displaying the classification report, gives precision, recall, f1-score for each class
print("\nAccuracy Score: ", accuracy_score(Y_test, Y_pred)) # Displaying the accuracy score of the model
