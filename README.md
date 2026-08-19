# Peatland Fire Risk Prediction System

## 📌 Deskripsi Proyek
Proyek ini merupakan **Sistem Prediksi Risiko Kebakaran Lahan Gambut** yang dikembangkan untuk mata kuliah Big Data. Tujuan utama dari proyek ini adalah memprediksi risiko kebakaran lahan gambut di wilayah Sumatera dan Kalimantan menggunakan data satelit, yang mencakup suhu permukaan lahan (MODIS LST), indeks vegetasi (MODIS NDVI), dan curah hujan (CHIRPS). 

Model ini memprediksi:
1. **Probabilitas Kebakaran (Classification)**: Apakah akan terjadi kebakaran.
2. **Waktu Menuju Kondisi Kering Kritis (Regression)**: Berapa hari hingga curah hujan turun di bawah batas kritis (4mm/hari).

Proyek ini menggunakan basis ilmiah bahwa risiko kebakaran meningkat secara non-linear ketika curah hujan berada di bawah 4mm/hari (Field et al., 2016).

---

## 🏗️ Arsitektur & Pipeline Data

Pipeline pemrosesan data terdiri dari empat fase utama:

### Fase 1: Data Akuisisi (Google Earth Engine)
- Mengagregasi citra satelit (LST, NDVI, Rainfall) pada musim kemarau (Juli - Oktober) selama periode 2015-2023.
- Menggunakan titik api NASA FIRMS (confidence >80%) sebagai label untuk kejadian kebakaran.
- Mengekstraksi fitur musiman per provinsi dengan tutupan lahan gambut tinggi (Riau, Sumatera Selatan, Jambi, Kalimantan Tengah, Kalimantan Barat, dll).

### Fase 2: Pemrosesan Data (PySpark)
- **Feature Engineering**: Membuat fitur-fitur turunan baru seperti `lst_anomaly`, `rainfall_deficit_30d`, dan one-hot encoding untuk provinsi (`province_vector`).
- **Temporal Train/Test Split**: Pembagian data dilakukan secara temporal untuk menghindari *data leakage*:
  - **Data Latih (Train)**: 2015 - 2021 (termasuk kejadian El Niño 2015 & 2019).
  - **Data Uji (Test)**: 2022 - 2023 (pengujian di masa depan yang tidak terlihat oleh model).
- Data latih dan uji disimpan dalam format Parquet untuk akses baca yang optimal.

### Fase 3: Pemodelan Machine Learning
Dua jenis model dilatih untuk menyelesaikan tugas prediktif:
- **Classifier**: Memprediksi kemunculan titik api (binary classification). Menggunakan SMOTE untuk menangani ketidakseimbangan kelas (imbalanced data).
- **Regressor**: Memprediksi jumlah hari menuju defisit hujan (days to 4mm breach).
- Model dievaluasi membandingkan 3 algoritma: **XGBoost, Random Forest, dan LightGBM**. Grid Search digunakan untuk *hyperparameter tuning*.

### Fase 4: Penilaian Risiko Distrik
Model yang telah dilatih digunakan untuk menghasilkan penilaian risiko agregat di tingkat provinsi, dan diklasifikasikan ke dalam berbagai kategori risiko (Moderate, High, Critical).

---

## 📊 Hasil dan Evaluasi Model

### 1. Performa Model Klasifikasi (Memprediksi Titik Api)
Berdasarkan uji pada data tahun 2022-2023:
| Model | ROC-AUC | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|---|
| **Random Forest** | 0.6313 | 0.0256 | 0.4210 | 0.0482 | 71.41% |
| **XGBoost** | 0.6089 | 0.0240 | 0.4561 | 0.0457 | 67.22% |
| **LightGBM** | 0.6285 | 0.0230 | 0.4736 | 0.0439 | 64.47% |

*Catatan: F1-score dan precision rendah merupakan karakteristik wajar karena data kebencanaan (titik api) secara inheren sangat tidak seimbang (imbalanced), dan pengujian murni pada data tahun berbeda (2022-2023).*

### 2. Performa Model Regresi (Memprediksi Hari menuju Defisit Hujan Kritis)
Berdasarkan uji pada data tahun 2022-2023:
| Model | MAE (Hari) | RMSE (Hari) | R² |
|---|---|---|---|
| **XGBoost** | 0.017 | 0.132 | 0.9989 |
| **Random Forest** | 0.044 | 0.146 | 0.9986 |
| **LightGBM** | 0.500 | 0.906 | 0.9487 |

Model XGBoost dan Random Forest sangat akurat dalam memprediksi jumlah hari menuju curah hujan kritis (< 4mm/hari) dengan *Mean Absolute Error* (MAE) kurang dari 1 hari.

### 3. Fitur Paling Penting (Feature Importance)
Model mengungkap bahwa lima variabel yang paling berpengaruh kuat terhadap kemunculan kebakaran hutan adalah:
1. `lst_max` (Suhu Permukaan Lahan Maksimum) - 20.5%
2. `mean_lst` (Rata-rata Suhu Permukaan Lahan) - 19.5%
3. `rainfall_deficit_30d` (Defisit Curah Hujan 30 Hari) - 15.3%
4. `mean_daily_rainfall` (Rata-rata Curah Hujan Harian) - 9.8%
5. `total_rainfall_season` (Total Hujan Satu Musim) - 9.7%

---

## 🚨 Kesimpulan Tingkat Risiko Provinsi (Output Akhir)

Sistem pada akhirnya memetakan probabilitas prediksi dan memproyeksikannya sebagai Tingkat Risiko Kebakaran Provinsi. Berdasarkan pengujian dari hasil prediksi model, lima provinsi dengan risiko paling kritis adalah:

1. **Sumatera Selatan** (Skor Risiko: 99.99 - **Kritis**)
2. **Kalimantan Barat** (Skor Risiko: 94.13 - **Kritis**)
3. **Riau** (Skor Risiko: 89.35 - **Kritis**)
4. **Jambi** (Skor Risiko: 87.60 - **Kritis**)
5. **Kalimantan Tengah** (Skor Risiko: 60.65 - **Tinggi**)

Hasil skor disimpulkan di dalam folder model dengan nama file format `district_risk_assessment_<model>.csv`

---

## 🛠️ Persyaratan Lingkungan (Requirements)
- Python 3.12+
- Google Colab (Dianjurkan untuk otentikasi Earth Engine dan Drive)
- PySpark 4.0.3 (Standalone / single-machine mode)
- Google Earth Engine API
- XGBoost, scikit-learn, LightGBM
- Pandas, Numpy, Matplotlib
