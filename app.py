
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get feature values from the form
        variance = float(request.form["variance"])
        skewness = float(request.form["skewness"])
        curtosis = float(request.form["curtosis"])
        entropy = float(request.form["entropy"])

        # Arrange the input features
        features = np.array(
            [[variance, skewness, curtosis, entropy]]
        )

        # Scale the features
        scaled_features = scaler.transform(features)

        # Predict the banknote class
        prediction = model.predict(scaled_features)[0]

        if prediction == 0:
            result = "Genuine Banknote"
        else:
            result = "Counterfeit Banknote"

        return render_template("index.html", prediction=result)

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            prediction="Please enter valid numerical values."
        )


if __name__ == "__main__":
    app.run(debug=True)
