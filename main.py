from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np


# ======================
# LOAD MODEL
# ======================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ======================
# FASTAPI APP
# ======================
app = FastAPI()

# ======================
# CORS (WAJIB BIAR HTML BISA AKSES)
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # aman buat UAS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# INPUT DATA SCHEMA
# ======================
class InputData(BaseModel):
    Year: float
    Engine_HP: float
    Engine_Cylinders: float
    highway_MPG: float
    city_mpg: float
    Popularity: float


# ======================
# ROOT
# ======================
@app.get("/")
def root():
    return {"message": "API Prediksi Harga Mobil Sport"}

# ======================
# PREDICT
# ======================
@app.post("/predict")
def predict(data: InputData):
    fitur = np.array([
        data.Year,
        data.Engine_HP,
        data.Engine_Cylinders,
        data.highway_MPG,
        data.city_mpg,
        data.Popularity
    ]).reshape(1, -1)

    prediksi = model.predict(fitur)
    print(fitur)
    print(prediksi)

    return {
        "predicted_price": round(float(prediksi[0]), 2)
    }
