#  UAS Machine Learning – Prediksi Harga Mobil Sport

Nama        : Azzrifa Nurul Aini
NPM         : 2441037
Mata Kuliah : Machine Learning
Jenis Tugas : Ujian Akhir Semester (UAS)

Proyek ini dibuat untuk memenuhi tugas **Ujian Akhir Semester (UAS)** mata kuliah Machine Learning.

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun model *Machine Learning* yang dapat memprediksi harga mobil sport (MSRP) berdasarkan beberapa fitur numerik kendaraan. Model dilatih menggunakan dataset publik dari Kaggle dan diimplementasikan dalam bentuk API menggunakan FastAPI, serta dilengkapi dengan frontend HTML sederhana untuk melakukan prediksi.


## Dataset
Sumber: Kaggle – *Car Features and MSRP Dataset*
Nama File: `data.csv`

### Fitur yang Digunakan:
1. Year (Tahun produksi)
2. Engine HP (Tenaga mesin)
3. Engine Cylinders (Jumlah silinder)
4. Highway MPG (Konsumsi BBM jalan tol)
5. City MPG (Konsumsi BBM dalam kota)
6. Popularity (Popularitas mobil)

### Target:
MSRP (Harga mobil)


## Tahapan Pengerjaan
1. Load Dataset
    Dataset dibaca menggunakan library **Pandas**.
2. Seleksi Fitur
    Hanya fitur numerik yang digunakan agar kompatibel dengan model regresi.
3. Penanganan Missing Value
    Missing value ditangani dengan metode **mean imputation**.
4. Pembagian Data
    Dataset dibagi menjadi:
    80% data training
    20% data testing
5. Training Model
    Model yang digunakan adalah **Linear Regression** dari *Scikit-learn*.
6. Evaluasi Model
    Evaluasi dilakukan menggunakan **Mean Squared Error (MSE)**.
7. Deployment Model
    Model yang telah dilatih disimpan dalam file `model.pkl` dan digunakan dalam **FastAPI** sebagai REST API.


## Implementasi API (FastAPI)

### Endpoint:
`GET /` → Mengecek status API
`POST /predict` → Melakukan prediksi harga mobil

### Contoh Input JSON:
{
  "Year": 2020,
  "Engine_HP": 350,
  "Engine_Cylinders": 6,
  "highway_MPG": 26,
  "city_mpg": 18,
  "Popularity": 3000
}

### Contoh Output:
{
  "predicted_price": 69350.47
}

## Frontend
Frontend dibuat menggunakan **HTML + JavaScript** sederhana yang terhubung langsung ke API FastAPI untuk menampilkan hasil prediksi harga mobil secara real-time.


## Teknologi yang Digunakan
* Python
* Pandas
* Scikit-learn
* FastAPI
* Uvicorn
* HTML & JavaScript


## Kesimpulan
Model Machine Learning berhasil dibuat dan diimplementasikan untuk memprediksi harga mobil sport. Sistem berjalan dengan baik mulai dari proses training, evaluasi, hingga deployment API dan integrasi frontend.
Proyek ini menunjukkan penerapan konsep Machine Learning secara end-to-end, mulai dari pengolahan data hingga implementasi ke aplikasi nyata.

