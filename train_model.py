
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
df = pd.read_csv("BankNote_Authentication.csv", header=None)

# 2. Assign column names
df.columns = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
    "class"
]

# 3. Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# 4. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Create and train Linear SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# 7. Predict test results
y_pred = model.predict(X_test)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("===================================")
print(" COUNTERFEIT BANKNOTE DETECTION")
print("===================================")
print("Algorithm: Linear Support Vector Machine")
print("Total dataset records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save the trained model and scaler
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as svm_model.pkl")
print("Scaler saved as scaler.pkl")
print("\nTraining completed successfully!")
