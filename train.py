import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# ==================================================
# 1. Load Dataset
# ==================================================
data = pd.read_csv("data.csv")

print("Kolom dataset:")
print(data.columns)

# ==================================================
# 2. Seleksi Fitur Numerik
# ==================================================
numeric_data = data.select_dtypes(include=["int64", "float64"])

# ==================================================
# 3. Penentuan Target dan Fitur
# ==================================================
X = numeric_data[
    [
        "Year",
        "Engine HP",
        "Engine Cylinders",
        "highway MPG",
        "city mpg",
        "Popularity"
    ]
]
y = numeric_data["MSRP"]

# ==================================================
# 4. Penanganan Missing Value
# ==================================================
X = X.fillna(X.mean())

# ==================================================
# 5. Pembagian Data Training & Testing
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================================================
# 6. Training Model (Linear Regression)
# ==================================================
model = LinearRegression()
model.fit(X_train, y_train)

# ==================================================
# 7. Evaluasi Model
# ==================================================
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# ==================================================
# 8. Simpan Model
# ==================================================
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model berhasil dilatih dan disimpan (model.pkl)")
