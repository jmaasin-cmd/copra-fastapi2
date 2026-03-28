from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# =========================
# 📦 Load trained ML models
# =========================
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
knn_model = joblib.load("knn_model.pkl")
log_model = joblib.load("logistic_model.pkl")


# =========================
# 📥 Input schema (JSON body)
# =========================
class InputData(BaseModel):
    moisture: float
    temperature: float
    rgb: int


# =========================
# 🏠 Home route
# =========================
@app.get("/")
def home():
    return {"message": "Copra Quality ML API running"}


# =========================
# 🔮 Prediction route 
# =========================
@app.post("/predict")
def predict(data: InputData):

    # 🐞 DEBUG: See incoming request data
    print("Incoming request:", data)

    try:

    df = pd.DataFrame([{
        "Moisture": data.moisture,
        "RGB Color": data.rgb,
        "Temperature": data.temperature
    }])

        # =========================
        # 🤖 ML Predictions
        # =========================
        results = {
            "SVM": str(svm_model.predict(df)[0]),
            "Random Forest": str(rf_model.predict(df)[0]),
            "KNN": str(knn_model.predict(df)[0]),
            "Logistic Regression": str(log_model.predict(df)[0])
        }

        return {
            "input": df.to_dict(orient="records")[0],
            "predictions": results
        }

    except Exception as e:
        return {"error": str(e)}
