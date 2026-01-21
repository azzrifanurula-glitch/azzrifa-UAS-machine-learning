from fastapi.responses import HTMLResponse
from fastapi import FastAPI
app = FastAPI(
    title="API Prediksi Harga Mobil Sport",
    description="""
    API ini digunakan untuk memprediksi harga mobil sport
    berdasarkan spesifikasi kendaraan menggunakan
    model Machine Learning (Linear Regression).
    
    Fitur input meliputi:
    - Tahun kendaraan
    - Engine Horsepower
    - Jumlah silinder
    - Highway MPG
    - City MPG
    - Popularity
    """,
    version="1.0"
)

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
app = FastAPI(
    title="Prediksi Harga Mobil Sport",
    description="API Machine Learning menggunakan FastAPI",
    version="1.0"
)

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Prediksi Harga Mobil Sport</title>
            <style>
                body {
                    font-family: Arial;
                    background-color: #f4f6f8;
                    text-align: center;
                    padding-top: 100px;
                }
                .card {
                    background: white;
                    display: inline-block;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
                a {
                    display: inline-block;
                    margin-top: 15px;
                    text-decoration: none;
                    color: white;
                    background: #2563eb;
                    padding: 10px 20px;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Prediksi Harga Mobil Sport</h1>
                <p>API berbasis Machine Learning menggunakan FastAPI</p>
                <a href="/docs">Buka Dokumentasi API</a>
            </div>
        </body>
    </html>
    """

# ======================
# CORS (WAJIB BIAR HTML BISA AKSES)
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
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
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

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
