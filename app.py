from flask import Flask, render_template, request
import pickle
from utils.feature_extractor import clean_text
from utils.risk_calculator import calculate_risk

app = Flask(__name__)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    risk = None
    reasons = []

    if request.method == "POST":
        text = request.form["job_description"]

        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        risk, reasons = calculate_risk(text)

        if pred == 1:
            result = f"🚨 Fake Job (ML Confidence: {prob*100:.2f}%)"
        else:
            result = f"✅ Real Job (Confidence: {(1-prob)*100:.2f}%)"

    return render_template("index.html", result=result, risk=risk, reasons=reasons)

if __name__ == "__main__":
    app.run(debug=True)
