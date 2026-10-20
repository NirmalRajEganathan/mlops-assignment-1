from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load(
    "artifacts/random_forest_model.pkl"
)


@app.get("/")
def home():

    return {
        "message": "Heart Disease Prediction API"
    }


@app.post("/predict")
def predict(
    age: float,
    sex: float,
    cp: float,
    trestbps: float,
    chol: float,
    fbs: float,
    restecg: float,
    thalach: float,
    exang: float,
    oldpeak: float,
    slope: float,
    ca: float,
    thal: float,
):

    data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    )

    prediction = model.predict(data)

    return {
        "prediction": int(prediction[0])
    }