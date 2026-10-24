# Counterfeit Banknote Detection Using Linear SVM

## Overview
This project uses a Linear Support Vector Machine (SVM) to classify banknotes as genuine or counterfeit based on numerical features extracted from banknote images.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- HTML and CSS
- Joblib

## Dataset
The project uses the UCI Banknote Authentication dataset. It contains four input features:
- Variance
- Skewness
- Kurtosis
- Entropy

The target column represents the banknote class.

## Machine Learning Workflow
1. Load and prepare the dataset.
2. Split the data into training and testing sets.
3. Standardize the numerical features using `StandardScaler`.
4. Train a Linear SVM classifier.
5. Evaluate the model using accuracy, a classification report, and a confusion matrix.
6. Save the trained model and scaler.
7. Use Flask to connect the model to a web interface.

## Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/tejswividhate-06/banknote-detection-using-linear-svm.git
cd banknote-detection-using-linear-svm
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
```

Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Start the web application
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Project Structure
```text
Banknote-Detection-Using-Linear-SVM/
├── app.py
├── train_model.py
├── BankNote_Authentication.csv
├── requirements.txt
├── svm_model.pkl
├── scaler.pkl
├── templates/
│   └── index.html
└── README.md
```

## Disclaimer
This project is intended for educational purposes. Its predictions are based on model inputs and should not be treated as definitive verification of physical currency.

## Author
Tejswi Nandkumar Vidhate
