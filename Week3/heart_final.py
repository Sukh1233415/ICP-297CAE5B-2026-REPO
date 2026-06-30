import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Dataset

df = pd.read_csv(r"Week2\heart.csv")
print("Dataset Loaded Successfully\n")
print(df.head())

# Check Missing Values

print(df.isnull().sum())
# Remove duplicates
df = df.drop_duplicates()

# Feature Selection

X = df.drop("target", axis=1)
y = df["target"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print("\nAccuracy :", accuracy)

# Classification Report
report = classification_report(
    y_test,
    prediction
)
print(report)
with open("classification_report.txt", "w") as f:
    f.write("Heart Disease Prediction\n\n")
    f.write(report)
print("classification_report.txt created")

# Confusion Matrix

cm = confusion_matrix(
    y_test,
    prediction
)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
print("confusion_matrix.png saved")

# Save Model
joblib.dump(model, "heart_model.pkl")
print("heart_model.pkl saved")