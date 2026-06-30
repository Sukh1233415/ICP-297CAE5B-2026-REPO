# Heart Disease Prediction Final Report

# Introduction
Heart disease causes most deaths around the globe, hence early diagnosis aids health care professionals in training them. Medical professionals can be supported by Macgine Learning models by leveraging patient health information data foor forecasting.

# Objective
To build a machine learning classification model that predicts whether a patient has heart disease based on medical features.

# Dataset

Dataset Name:
heart.csv
Number of Records:
303
Target Variable:
target
Target Values
0 = No Heart Disease
1 = Heart Disease

# Features Used
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- ST Depression
- Slope
- Number of Major Vessels
- Thalassemia

# Data Preprocessing
The following preprocessing steps were performed:
- Loaded dataset using Pandas.
- Checked missing values.
- Removed duplicate records.
- Selected input and target variables.
- Split data into training and testing datasets.

# Machine Learning Algorithm
Random Forest Classifier
Reason for selection:
- High prediction accuracy
- Handles non-linear relationships
- Less prone to overfitting
- Works well on medical datasets

# Evaluation Metrics
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

# Results
The Random Forest model achieved high classification accuracy and successfully classified patients into heart disease and non-heart disease categories.
The confusion matrix showed good prediction performance.

# Model Saving

The trained model was saved as
heart_model.pkl
The classification report was saved as
classification_report.txt
The confusion matrix was saved as
confusion_matrix.png

# Applications
- Hospital diagnosis support
- Health monitoring systems
- Medical decision support
- Healthcare analytics

# Conclusion
The Heart Disease Prediction model successfully predicts whether a patient is likely to have heart disease. The project demonstrates the complete machine learning workflow from data preprocessing to model evaluation and model serialization.

# Future Improvements
- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Flask web deployment
- Streamlit dashboard