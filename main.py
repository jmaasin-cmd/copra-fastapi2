from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load models
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
knn_model = joblib.load("knn_model.pkl")
log_model = joblib.load("logistic_model.pkl")

# ✅ Define request body
class InputData(BaseModel):
    moisture: float
    temperature: float
    rgb: int


@app.get("/")
def home():
    return {"message": "Copra Quality ML API running"}


@app.post("/predict")
def predict(data: InputData):

    try:
        values = np.array([[data.moisture, data.temperature, data.rgb]])

        results = {
            "SVM": str(svm_model.predict(values)[0]),
            "Random Forest": str(rf_model.predict(values)[0]),
            "KNN": str(knn_model.predict(values)[0]),
            "Logistic Regression": str(log_model.predict(values)[0])
        }

        return results

    except Exception as e:
        return {"error": str(e)}
