from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load Trained Model
model = joblib.load("model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Page
@app.route("/predict")
def predict_page():
    return render_template("predict.html")


# Prediction Result
@app.route("/result", methods=["POST"])
def result():

    age = int(request.form["age"])
    income = float(request.form["income"])
    credit = int(request.form["credit"])
    loan = float(request.form["loan"])

    prediction = model.predict([[age, income, credit, loan]])

    if prediction[0] == 1:
        result = "✅ Loan Approved"
    else:
        result = "❌ Loan Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)