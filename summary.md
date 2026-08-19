# peatland_fire_prediction.ipynb

# Peatland Fire Risk Prediction System

**Project:** Forest Fire Detection - Big Data Course  
**Goal:** Predict peatland fire risk in Sumatra and Kalimantan using satellite data (MODIS LST/NDVI + CHIRPS rainfall)

**Pipeline:**
1. Earth Engine: Aggregate 10TB satellite imagery → seasonal features (~7M rows)
2. PySpark: Preprocess features, train/test split (2015-2021 | 2022-2023)
3. XGBoost: Dual models (fire probability + days to critical drought)
4. Output: District-level risk CSV

**Scientific Foundation:**
- Field et al. (2016): Fire risk nonlinearly increases when rainfall <4mm/day
- Nurdiati et al. (2024): ML methods for Kalimantan fire prediction
- Prayoga et al. (2024): Threshold-based early warning for Riau

✓ Running in Google Colab
✓ Python 3.12.13

✓ Dependencies installed

✓ All imports successful

## Phase 1: Environment Setup

### Earth Engine Authentication

✓ Earth Engine already authenticated
✓ Earth Engine initialized (test elevation range: [1296001, 417601])

### Google Drive Mount

Data will be stored in `MyDrive/forest-fire-data/`:
- `raw/` - Earth Engine exports (parquet files)
- `processed/` - PySpark outputs (train/test splits)
- `models/` - Trained XGBoost models (Phase 3)

Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
✓ Data directory: /content/drive/MyDrive/forest-fire-data
  - raw/: Earth Engine exports
  - processed/: PySpark outputs
  - models/: Trained models

✓ /content/drive/MyDrive/forest-fire-data/raw
✓ /content/drive/MyDrive/forest-fire-data/processed
✓ /content/drive/MyDrive/forest-fire-data/models

### Load Peatland Province Boundaries

Filter to high-peat provinces: Riau, Sumatra Selatan, Jambi, Kalimantan Tengah, Kalimantan Barat

✓ Loaded 100 province-year combinations
  Years: [np.int64(2015), np.int64(2016), np.int64(2017), np.int64(2018), np.int64(2019), np.int64(2020), np.int64(2021), np.int64(2022), np.int64(2023), np.int64(2024)]
  Provinces: ['ACEH', 'JAMBI', 'KALIMANTAN BARAT', 'KALIMANTAN TENGAH', 'KALIMANTAN TIMUR', 'KALIMANTAN UTARA', 'RIAU', 'SUMATERA BARAT', 'SUMATERA SELATAN', 'SUMATERA UTARA']

Top 5 provinces by peatland area (2019):
  - RIAU: 355.0K ha
  - KALIMANTAN TENGAH: 254.9K ha
  - KALIMANTAN BARAT: 154.6K ha
  - SUMATERA SELATAN: 111.6K ha
  - JAMBI: 49.0K ha
✓ All imports successful

### Test Export: Riau 2019 Dry Season

Test Earth Engine aggregation on single province (Riau) and single year (2019) to verify:
1. Data loading works
2. Seasonal aggregation logic correct
3. Export to Drive successful

Test export function defined

→ Loading data for RIAU, 2019-07-01 to 2019-10-30...
  ✓ LST images: 242
  ✓ NDVI images: 7
  ✓ Rainfall images: 121
→ Sampling 668 grid points...
✓ Exported 668 rows × 11 columns

Sample data:
   consecutive_dry_days  latitude   longitude  lst_max  mean_daily_rainfall  \
0                    68  1.961224  103.054174    38.21             4.566635   
1                    74  0.633018  100.417473    34.29             4.679079   
2                    62  1.912082  100.040259    35.13             6.108468   
3                    72  0.609443  101.637205    38.35             3.970790   
4                    82 -0.582406  100.563600    33.23             3.401129   

    mean_lst  min_ndvi province           season  total_rainfall_season  year  
0  31.817200    0.4024     RIAU  2019-dry-season             552.562785  2019  
1  29.422222    0.2508     RIAU  2019-dry-season             566.168582  2019  
2  31.256667    0.3263     RIAU  2019-dry-season             739.124581  2019  
3  30.870000    0.4840     RIAU  2019-dry-season             480.465605  2019  
4  27.046000    0.0838     RIAU  2019-dry-season             411.536608  2019  

Column summary:
       consecutive_dry_days    latitude   longitude     lst_max  \
count            668.000000  668.000000  668.000000  668.000000   
mean              78.950599    0.623293  101.924956   35.028204   
std                9.560928    1.005515    1.205318    2.859455   
min               55.000000   -0.995098  100.005279   24.150000   
25%               72.000000   -0.263681  100.906142   33.205000   
50%               79.000000    0.543183  101.869300   34.840000   
75%               86.000000    1.486176  102.975537   36.580000   
max              100.000000    2.488239  104.738910   46.770000   

       mean_daily_rainfall    mean_lst    min_ndvi  total_rainfall_season  \
count           668.000000  668.000000  668.000000             668.000000   
mean              3.972195   30.082266    0.375330             480.635590   
std               1.056959    2.096331    0.230686             127.892098   
min               1.991199   20.182857   -0.050700             240.935053   
25%               3.128880   29.021833    0.176500             378.594441   
50%               3.819899   30.093559    0.339150             462.207802   
75%               4.632105   31.069480    0.561150             560.484756   
max               7.213615   38.233077    0.853800             872.847482   

         year  
count   668.0  
mean   2019.0  
std       0.0  
min    2019.0  
25%    2019.0  
50%    2019.0  
75%    2019.0  
max    2019.0  

✓ Saved to /content/drive/MyDrive/forest-fire-data/test_export_riau_2019.csv

✓ Verified: 668 rows loaded from disk
  Columns: consecutive_dry_days, latitude, longitude, lst_max, mean_daily_rainfall, mean_lst...
✓ No missing values

## Phase 2: Full Data Pipeline

### FIRMS Fire Labels

Load NASA FIRMS thermal hotspot data to label fire occurrences (confidence >80%).

### Full Aggregation Function

Process all provinces and all years (2015-2023) with:
- Seasonal aggregation (July-October dry season)
- Fire labels from FIRMS
- Days to 4mm/day breach calculation

      (DEBUG: FIRMS images 2019: 122 total detections)
✓ FIRMS data accessible for 2019


→ Processing year 2015...
  → Processing ACEH (ID-11) for 2015...
    (DEBUG: Entering aggregate_province_year_features for ACEH, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 165 features for ACEH, 2015)
    ✓ Successfully retrieved 165 rows
  → Processing JAMBI (ID-15) for 2015...
    (DEBUG: Entering aggregate_province_year_features for JAMBI, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 443 features for JAMBI, 2015)
    ✓ Successfully retrieved 443 rows
  → Processing KALIMANTAN BARAT (ID-61) for 2015...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN BARAT, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 345 features for KALIMANTAN BARAT, 2015)
    ✓ Successfully retrieved 345 rows
  → Processing KALIMANTAN TENGAH (ID-62) for 2015...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TENGAH, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 412 features for KALIMANTAN TENGAH, 2015)
    ✓ Successfully retrieved 412 rows
  → Processing KALIMANTAN TIMUR (ID-64) for 2015...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TIMUR, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 275 features for KALIMANTAN TIMUR, 2015)
    ✓ Successfully retrieved 275 rows
  → Processing KALIMANTAN UTARA (ID-65) for 2015...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN UTARA, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 433 features for KALIMANTAN UTARA, 2015)
    ✓ Successfully retrieved 433 rows
  → Processing RIAU (ID-14) for 2015...
    (DEBUG: Entering aggregate_province_year_features for RIAU, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 331 features for RIAU, 2015)
    ✓ Successfully retrieved 331 rows
  → Processing SUMATERA BARAT (ID-13) for 2015...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA BARAT, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 190 features for SUMATERA BARAT, 2015)
    ✓ Successfully retrieved 190 rows
  → Processing SUMATERA SELATAN (ID-16) for 2015...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA SELATAN, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 403 features for SUMATERA SELATAN, 2015)
    ✓ Successfully retrieved 403 rows
  → Processing SUMATERA UTARA (ID-12) for 2015...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA UTARA, 2015)
      (DEBUG: FIRMS images 2015: 122 total detections)
    (DEBUG: Earth Engine returned 313 features for SUMATERA UTARA, 2015)
    ✓ Successfully retrieved 313 rows
  ✓ Saved 3310 rows to seasonal_features_2015.parquet

→ Processing year 2016...
  → Processing ACEH (ID-11) for 2016...
    (DEBUG: Entering aggregate_province_year_features for ACEH, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 165 features for ACEH, 2016)
    ✓ Successfully retrieved 165 rows
  → Processing JAMBI (ID-15) for 2016...
    (DEBUG: Entering aggregate_province_year_features for JAMBI, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 443 features for JAMBI, 2016)
    ✓ Successfully retrieved 443 rows
  → Processing KALIMANTAN BARAT (ID-61) for 2016...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN BARAT, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 345 features for KALIMANTAN BARAT, 2016)
    ✓ Successfully retrieved 345 rows
  → Processing KALIMANTAN TENGAH (ID-62) for 2016...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TENGAH, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 412 features for KALIMANTAN TENGAH, 2016)
    ✓ Successfully retrieved 412 rows
  → Processing KALIMANTAN TIMUR (ID-64) for 2016...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TIMUR, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 275 features for KALIMANTAN TIMUR, 2016)
    ✓ Successfully retrieved 275 rows
  → Processing KALIMANTAN UTARA (ID-65) for 2016...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN UTARA, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 433 features for KALIMANTAN UTARA, 2016)
    ✓ Successfully retrieved 433 rows
  → Processing RIAU (ID-14) for 2016...
    (DEBUG: Entering aggregate_province_year_features for RIAU, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 331 features for RIAU, 2016)
    ✓ Successfully retrieved 331 rows
  → Processing SUMATERA BARAT (ID-13) for 2016...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA BARAT, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 190 features for SUMATERA BARAT, 2016)
    ✓ Successfully retrieved 190 rows
  → Processing SUMATERA SELATAN (ID-16) for 2016...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA SELATAN, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 403 features for SUMATERA SELATAN, 2016)
    ✓ Successfully retrieved 403 rows
  → Processing SUMATERA UTARA (ID-12) for 2016...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA UTARA, 2016)
      (DEBUG: FIRMS images 2016: 122 total detections)
    (DEBUG: Earth Engine returned 313 features for SUMATERA UTARA, 2016)
    ✓ Successfully retrieved 313 rows
  ✓ Saved 3310 rows to seasonal_features_2016.parquet

→ Processing year 2017...
  → Processing ACEH (ID-11) for 2017...
    (DEBUG: Entering aggregate_province_year_features for ACEH, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 165 features for ACEH, 2017)
    ✓ Successfully retrieved 165 rows
  → Processing JAMBI (ID-15) for 2017...
    (DEBUG: Entering aggregate_province_year_features for JAMBI, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 443 features for JAMBI, 2017)
    ✓ Successfully retrieved 443 rows
  → Processing KALIMANTAN BARAT (ID-61) for 2017...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN BARAT, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 345 features for KALIMANTAN BARAT, 2017)
    ✓ Successfully retrieved 345 rows
  → Processing KALIMANTAN TENGAH (ID-62) for 2017...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TENGAH, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 411 features for KALIMANTAN TENGAH, 2017)
    ✓ Successfully retrieved 411 rows
  → Processing KALIMANTAN TIMUR (ID-64) for 2017...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TIMUR, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 275 features for KALIMANTAN TIMUR, 2017)
    ✓ Successfully retrieved 275 rows
  → Processing KALIMANTAN UTARA (ID-65) for 2017...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN UTARA, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 433 features for KALIMANTAN UTARA, 2017)
    ✓ Successfully retrieved 433 rows
  → Processing RIAU (ID-14) for 2017...
    (DEBUG: Entering aggregate_province_year_features for RIAU, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 331 features for RIAU, 2017)
    ✓ Successfully retrieved 331 rows
  → Processing SUMATERA BARAT (ID-13) for 2017...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA BARAT, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 190 features for SUMATERA BARAT, 2017)
    ✓ Successfully retrieved 190 rows
  → Processing SUMATERA SELATAN (ID-16) for 2017...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA SELATAN, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 403 features for SUMATERA SELATAN, 2017)
    ✓ Successfully retrieved 403 rows
  → Processing SUMATERA UTARA (ID-12) for 2017...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA UTARA, 2017)
      (DEBUG: FIRMS images 2017: 122 total detections)
    (DEBUG: Earth Engine returned 313 features for SUMATERA UTARA, 2017)
    ✓ Successfully retrieved 313 rows
  ✓ Saved 3309 rows to seasonal_features_2017.parquet

→ Processing year 2018...
  → Processing ACEH (ID-11) for 2018...
    (DEBUG: Entering aggregate_province_year_features for ACEH, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 165 features for ACEH, 2018)
    ✓ Successfully retrieved 165 rows
  → Processing JAMBI (ID-15) for 2018...
    (DEBUG: Entering aggregate_province_year_features for JAMBI, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 443 features for JAMBI, 2018)
    ✓ Successfully retrieved 443 rows
  → Processing KALIMANTAN BARAT (ID-61) for 2018...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN BARAT, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 345 features for KALIMANTAN BARAT, 2018)
    ✓ Successfully retrieved 345 rows
  → Processing KALIMANTAN TENGAH (ID-62) for 2018...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TENGAH, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 412 features for KALIMANTAN TENGAH, 2018)
    ✓ Successfully retrieved 412 rows
  → Processing KALIMANTAN TIMUR (ID-64) for 2018...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TIMUR, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 275 features for KALIMANTAN TIMUR, 2018)
    ✓ Successfully retrieved 275 rows
  → Processing KALIMANTAN UTARA (ID-65) for 2018...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN UTARA, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 433 features for KALIMANTAN UTARA, 2018)
    ✓ Successfully retrieved 433 rows
  → Processing RIAU (ID-14) for 2018...
    (DEBUG: Entering aggregate_province_year_features for RIAU, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 331 features for RIAU, 2018)
    ✓ Successfully retrieved 331 rows
  → Processing SUMATERA BARAT (ID-13) for 2018...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA BARAT, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 190 features for SUMATERA BARAT, 2018)
    ✓ Successfully retrieved 190 rows
  → Processing SUMATERA SELATAN (ID-16) for 2018...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA SELATAN, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 403 features for SUMATERA SELATAN, 2018)
    ✓ Successfully retrieved 403 rows
  → Processing SUMATERA UTARA (ID-12) for 2018...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA UTARA, 2018)
      (DEBUG: FIRMS images 2018: 120 total detections)
    (DEBUG: Earth Engine returned 313 features for SUMATERA UTARA, 2018)
    ✓ Successfully retrieved 313 rows
  ✓ Saved 3310 rows to seasonal_features_2018.parquet

→ Processing year 2019...
  → Processing ACEH (ID-11) for 2019...
    (DEBUG: Entering aggregate_province_year_features for ACEH, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 165 features for ACEH, 2019)
    ✓ Successfully retrieved 165 rows
  → Processing JAMBI (ID-15) for 2019...
    (DEBUG: Entering aggregate_province_year_features for JAMBI, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 443 features for JAMBI, 2019)
    ✓ Successfully retrieved 443 rows
  → Processing KALIMANTAN BARAT (ID-61) for 2019...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN BARAT, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 345 features for KALIMANTAN BARAT, 2019)
    ✓ Successfully retrieved 345 rows
  → Processing KALIMANTAN TENGAH (ID-62) for 2019...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TENGAH, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 412 features for KALIMANTAN TENGAH, 2019)
    ✓ Successfully retrieved 412 rows
  → Processing KALIMANTAN TIMUR (ID-64) for 2019...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TIMUR, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 275 features for KALIMANTAN TIMUR, 2019)
    ✓ Successfully retrieved 275 rows
  → Processing KALIMANTAN UTARA (ID-65) for 2019...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN UTARA, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 433 features for KALIMANTAN UTARA, 2019)
    ✓ Successfully retrieved 433 rows
  → Processing RIAU (ID-14) for 2019...
    (DEBUG: Entering aggregate_province_year_features for RIAU, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 331 features for RIAU, 2019)
    ✓ Successfully retrieved 331 rows
  → Processing SUMATERA BARAT (ID-13) for 2019...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA BARAT, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 190 features for SUMATERA BARAT, 2019)
    ✓ Successfully retrieved 190 rows
  → Processing SUMATERA SELATAN (ID-16) for 2019...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA SELATAN, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 403 features for SUMATERA SELATAN, 2019)
    ✓ Successfully retrieved 403 rows
  → Processing SUMATERA UTARA (ID-12) for 2019...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA UTARA, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 313 features for SUMATERA UTARA, 2019)
    ✓ Successfully retrieved 313 rows
  ✓ Saved 3310 rows to seasonal_features_2019.parquet

→ Processing year 2020...
  → Processing ACEH (ID-11) for 2020...
    (DEBUG: Entering aggregate_province_year_features for ACEH, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 165 features for ACEH, 2020)
    ✓ Successfully retrieved 165 rows
  → Processing JAMBI (ID-15) for 2020...
    (DEBUG: Entering aggregate_province_year_features for JAMBI, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 443 features for JAMBI, 2020)
    ✓ Successfully retrieved 443 rows
  → Processing KALIMANTAN BARAT (ID-61) for 2020...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN BARAT, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 345 features for KALIMANTAN BARAT, 2020)
    ✓ Successfully retrieved 345 rows
  → Processing KALIMANTAN TENGAH (ID-62) for 2020...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TENGAH, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 412 features for KALIMANTAN TENGAH, 2020)
    ✓ Successfully retrieved 412 rows
  → Processing KALIMANTAN TIMUR (ID-64) for 2020...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TIMUR, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 275 features for KALIMANTAN TIMUR, 2020)
    ✓ Successfully retrieved 275 rows
  → Processing KALIMANTAN UTARA (ID-65) for 2020...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN UTARA, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 433 features for KALIMANTAN UTARA, 2020)
    ✓ Successfully retrieved 433 rows
  → Processing RIAU (ID-14) for 2020...
    (DEBUG: Entering aggregate_province_year_features for RIAU, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 331 features for RIAU, 2020)
    ✓ Successfully retrieved 331 rows
  → Processing SUMATERA BARAT (ID-13) for 2020...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA BARAT, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 190 features for SUMATERA BARAT, 2020)
    ✓ Successfully retrieved 190 rows
  → Processing SUMATERA SELATAN (ID-16) for 2020...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA SELATAN, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 403 features for SUMATERA SELATAN, 2020)
    ✓ Successfully retrieved 403 rows
  → Processing SUMATERA UTARA (ID-12) for 2020...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA UTARA, 2020)
      (DEBUG: FIRMS images 2020: 122 total detections)
    (DEBUG: Earth Engine returned 313 features for SUMATERA UTARA, 2020)
    ✓ Successfully retrieved 313 rows
  ✓ Saved 3310 rows to seasonal_features_2020.parquet

→ Processing year 2021...
  → Processing ACEH (ID-11) for 2021...
    (DEBUG: Entering aggregate_province_year_features for ACEH, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 165 features for ACEH, 2021)
    ✓ Successfully retrieved 165 rows
  → Processing JAMBI (ID-15) for 2021...
    (DEBUG: Entering aggregate_province_year_features for JAMBI, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 443 features for JAMBI, 2021)
    ✓ Successfully retrieved 443 rows
  → Processing KALIMANTAN BARAT (ID-61) for 2021...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN BARAT, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 345 features for KALIMANTAN BARAT, 2021)
    ✓ Successfully retrieved 345 rows
  → Processing KALIMANTAN TENGAH (ID-62) for 2021...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TENGAH, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 412 features for KALIMANTAN TENGAH, 2021)
    ✓ Successfully retrieved 412 rows
  → Processing KALIMANTAN TIMUR (ID-64) for 2021...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TIMUR, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 275 features for KALIMANTAN TIMUR, 2021)
    ✓ Successfully retrieved 275 rows
  → Processing KALIMANTAN UTARA (ID-65) for 2021...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN UTARA, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 433 features for KALIMANTAN UTARA, 2021)
    ✓ Successfully retrieved 433 rows
  → Processing RIAU (ID-14) for 2021...
    (DEBUG: Entering aggregate_province_year_features for RIAU, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 331 features for RIAU, 2021)
    ✓ Successfully retrieved 331 rows
  → Processing SUMATERA BARAT (ID-13) for 2021...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA BARAT, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 190 features for SUMATERA BARAT, 2021)
    ✓ Successfully retrieved 190 rows
  → Processing SUMATERA SELATAN (ID-16) for 2021...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA SELATAN, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 403 features for SUMATERA SELATAN, 2021)
    ✓ Successfully retrieved 403 rows
  → Processing SUMATERA UTARA (ID-12) for 2021...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA UTARA, 2021)
      (DEBUG: FIRMS images 2021: 122 total detections)
    (DEBUG: Earth Engine returned 313 features for SUMATERA UTARA, 2021)
    ✓ Successfully retrieved 313 rows
  ✓ Saved 3310 rows to seasonal_features_2021.parquet

→ Processing year 2022...
  → Processing ACEH (ID-11) for 2022...
    (DEBUG: Entering aggregate_province_year_features for ACEH, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 165 features for ACEH, 2022)
    ✓ Successfully retrieved 165 rows
  → Processing JAMBI (ID-15) for 2022...
    (DEBUG: Entering aggregate_province_year_features for JAMBI, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 443 features for JAMBI, 2022)
    ✓ Successfully retrieved 443 rows
  → Processing KALIMANTAN BARAT (ID-61) for 2022...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN BARAT, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 345 features for KALIMANTAN BARAT, 2022)
    ✓ Successfully retrieved 345 rows
  → Processing KALIMANTAN TENGAH (ID-62) for 2022...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TENGAH, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 412 features for KALIMANTAN TENGAH, 2022)
    ✓ Successfully retrieved 412 rows
  → Processing KALIMANTAN TIMUR (ID-64) for 2022...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TIMUR, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 275 features for KALIMANTAN TIMUR, 2022)
    ✓ Successfully retrieved 275 rows
  → Processing KALIMANTAN UTARA (ID-65) for 2022...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN UTARA, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 433 features for KALIMANTAN UTARA, 2022)
    ✓ Successfully retrieved 433 rows
  → Processing RIAU (ID-14) for 2022...
    (DEBUG: Entering aggregate_province_year_features for RIAU, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 331 features for RIAU, 2022)
    ✓ Successfully retrieved 331 rows
  → Processing SUMATERA BARAT (ID-13) for 2022...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA BARAT, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 190 features for SUMATERA BARAT, 2022)
    ✓ Successfully retrieved 190 rows
  → Processing SUMATERA SELATAN (ID-16) for 2022...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA SELATAN, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 403 features for SUMATERA SELATAN, 2022)
    ✓ Successfully retrieved 403 rows
  → Processing SUMATERA UTARA (ID-12) for 2022...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA UTARA, 2022)
      (DEBUG: FIRMS images 2022: 122 total detections)
    (DEBUG: Earth Engine returned 313 features for SUMATERA UTARA, 2022)
    ✓ Successfully retrieved 313 rows
  ✓ Saved 3310 rows to seasonal_features_2022.parquet

→ Processing year 2023...
  → Processing ACEH (ID-11) for 2023...
    (DEBUG: Entering aggregate_province_year_features for ACEH, 2023)
      (DEBUG: Empty collection for mean_lst in ACEH, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in ACEH, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in ACEH, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for ACEH, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for ACEH, 2023.
  → Processing JAMBI (ID-15) for 2023...
    (DEBUG: Entering aggregate_province_year_features for JAMBI, 2023)
      (DEBUG: Empty collection for mean_lst in JAMBI, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in JAMBI, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in JAMBI, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for JAMBI, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for JAMBI, 2023.
  → Processing KALIMANTAN BARAT (ID-61) for 2023...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN BARAT, 2023)
      (DEBUG: Empty collection for mean_lst in KALIMANTAN BARAT, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in KALIMANTAN BARAT, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in KALIMANTAN BARAT, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for KALIMANTAN BARAT, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for KALIMANTAN BARAT, 2023.
  → Processing KALIMANTAN TENGAH (ID-62) for 2023...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TENGAH, 2023)
      (DEBUG: Empty collection for mean_lst in KALIMANTAN TENGAH, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in KALIMANTAN TENGAH, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in KALIMANTAN TENGAH, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for KALIMANTAN TENGAH, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for KALIMANTAN TENGAH, 2023.
  → Processing KALIMANTAN TIMUR (ID-64) for 2023...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN TIMUR, 2023)
      (DEBUG: Empty collection for mean_lst in KALIMANTAN TIMUR, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in KALIMANTAN TIMUR, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in KALIMANTAN TIMUR, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for KALIMANTAN TIMUR, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for KALIMANTAN TIMUR, 2023.
  → Processing KALIMANTAN UTARA (ID-65) for 2023...
    (DEBUG: Entering aggregate_province_year_features for KALIMANTAN UTARA, 2023)
      (DEBUG: Empty collection for mean_lst in KALIMANTAN UTARA, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in KALIMANTAN UTARA, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in KALIMANTAN UTARA, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for KALIMANTAN UTARA, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for KALIMANTAN UTARA, 2023.
  → Processing RIAU (ID-14) for 2023...
    (DEBUG: Entering aggregate_province_year_features for RIAU, 2023)
      (DEBUG: Empty collection for mean_lst in RIAU, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in RIAU, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in RIAU, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for RIAU, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for RIAU, 2023.
  → Processing SUMATERA BARAT (ID-13) for 2023...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA BARAT, 2023)
      (DEBUG: Empty collection for mean_lst in SUMATERA BARAT, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in SUMATERA BARAT, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in SUMATERA BARAT, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for SUMATERA BARAT, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for SUMATERA BARAT, 2023.
  → Processing SUMATERA SELATAN (ID-16) for 2023...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA SELATAN, 2023)
      (DEBUG: Empty collection for mean_lst in SUMATERA SELATAN, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in SUMATERA SELATAN, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in SUMATERA SELATAN, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for SUMATERA SELATAN, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for SUMATERA SELATAN, 2023.
  → Processing SUMATERA UTARA (ID-12) for 2023...
    (DEBUG: Entering aggregate_province_year_features for SUMATERA UTARA, 2023)
      (DEBUG: Empty collection for mean_lst in SUMATERA UTARA, 2023. Returning masked image.)
      (DEBUG: Empty collection for lst_max in SUMATERA UTARA, 2023. Returning masked image.)
      (DEBUG: Empty collection for min_ndvi in SUMATERA UTARA, 2023. Returning masked image.)
      (DEBUG: FIRMS images 2023: 122 total detections)
    (DEBUG: Earth Engine returned 0 features for SUMATERA UTARA, 2023)
    ⚠ `aggregate_province_year_features` returned an empty DataFrame for SUMATERA UTARA, 2023.
  ✗ No non-empty data exported for 2023. `year_data` was empty after processing all provinces.

✓ Export complete: 26479 total rows across 8 years

✓ Full export complete
  Files saved to: /content/drive/MyDrive/forest-fire-data/raw/

**Uncomment below to run full export (all years 2015-2023):**

→ Test export: Single year (2019)
  → ACEH...     (DEBUG: Entering aggregate_province_year_features for ACEH, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 165 features for ACEH, 2019)
✓ 165 rows
  → JAMBI...     (DEBUG: Entering aggregate_province_year_features for JAMBI, 2019)
      (DEBUG: FIRMS images 2019: 122 total detections)
    (DEBUG: Earth Engine returned 443 features for JAMBI, 2019)
✓ 443 rows

✓ Test successful: 608 rows exported
  Columns: ['consecutive_dry_days', 'days_since_season_start', 'days_to_4mm_breach', 'fire_occurred', 'latitude', 'longitude', 'lst_max', 'mean_daily_rainfall', 'mean_lst', 'min_ndvi', 'province', 'province_id', 'season', 'season_year', 'total_rainfall_season', 'year']

Sample:
   consecutive_dry_days  days_since_season_start  days_to_4mm_breach  \
0                    69                       60                  34   
1                    46                       60                  23   
2                    47                       60                  23   

   fire_occurred  latitude  longitude  lst_max  mean_daily_rainfall  \
0              0  5.028660  97.609669    33.71             4.104456   
1              1  4.801012  96.137420    31.63             7.826908   
2              0  4.001861  96.484629    33.83             7.032654   

    mean_lst  min_ndvi province province_id           season  season_year  \
0  31.463684    0.0662     ACEH       ID-11  2019-dry-season         2019   
1  27.309167    0.2022     ACEH       ID-11  2019-dry-season         2019   
2  30.372500    0.6180     ACEH       ID-11  2019-dry-season         2019   

   total_rainfall_season  year  
0             500.743600  2019  
1             954.882729  2019  
2             857.983758  2019  

**⚠ WARNING:** Full export takes 1-3 hours. Progress is saved per-year.

For faster testing, run single year first:

### PySpark Initialization

Standalone mode (single-machine, satisfies "harus pake pyspark" requirement).

✓ Spark session initialized
  Version: 4.0.3
  Master: local[*]
  Driver memory: 4GB

Data loading function defined

✓ Test data loaded: 668 rows

Schema:
root
 |-- consecutive_dry_days: integer (nullable = true)
 |-- latitude: double (nullable = true)
 |-- longitude: double (nullable = true)
 |-- lst_max: double (nullable = true)
 |-- mean_daily_rainfall: double (nullable = true)
 |-- mean_lst: double (nullable = true)
 |-- min_ndvi: double (nullable = true)
 |-- province: string (nullable = true)
 |-- season: string (nullable = true)
 |-- total_rainfall_season: double (nullable = true)
 |-- year: integer (nullable = true)


Sample:
+--------------------+------------------+------------------+------------------+-------------------+------------------+-------------------+--------+---------------+---------------------+----+
|consecutive_dry_days|          latitude|         longitude|           lst_max|mean_daily_rainfall|          mean_lst|           min_ndvi|province|         season|total_rainfall_season|year|
+--------------------+------------------+------------------+------------------+-------------------+------------------+-------------------+--------+---------------+---------------------+----+
|                  68| 1.961224163594329|103.05417418404947|38.210000000000036|  4.566634654998779|31.817200000000014|0.40240000000000004|    RIAU|2019-dry-season|     552.562784910202|2019|
|                  74|0.6330177519361887|100.41747261028033| 34.29000000000002|  4.679079055786133|29.422222222222274|             0.2508|    RIAU|2019-dry-season|    566.1685819625854|2019|
|                  62| 1.912081680255871|100.04025910860231| 35.13000000000005| 6.1084675788879395|31.256666666666717|0.32630000000000003|    RIAU|2019-dry-season|    739.1245813369751|2019|
+--------------------+------------------+------------------+------------------+-------------------+------------------+-------------------+--------+---------------+---------------------+----+
only showing top 3 rows

### Feature Engineering

Add derived features:
- One-hot encode provinces
- Rainfall anomalies
- LST anomalies

Feature engineering function defined

→ Engineering features...
✓ Feature engineering complete
  New columns: lst_anomaly, rainfall_deficit_30d, province_vector

Engineered schema:
root
 |-- province: string (nullable = true)
 |-- consecutive_dry_days: long (nullable = true)
 |-- days_since_season_start: long (nullable = true)
 |-- days_to_4mm_breach: long (nullable = true)
 |-- fire_occurred: integer (nullable = true)
 |-- latitude: double (nullable = true)
 |-- longitude: double (nullable = true)
 |-- lst_max: double (nullable = true)
 |-- mean_daily_rainfall: double (nullable = true)
 |-- mean_lst: double (nullable = true)
 |-- min_ndvi: double (nullable = true)
 |-- province_id: string (nullable = true)
 |-- season: string (nullable = true)
 |-- season_year: long (nullable = true)
 |-- total_rainfall_season: double (nullable = true)
 |-- year: long (nullable = true)
 |-- province_baseline_lst: double (nullable = true)
 |-- province_baseline_rainfall: double (nullable = true)
 |-- lst_anomaly: double (nullable = true)
 |-- rainfall_deficit_30d: double (nullable = true)
 |-- province_index: double (nullable = false)
 |-- province_vector: vector (nullable = true)


Sample with new features:
+--------+----+------------------+-------------------+--------------------+-------------+
|province|year|          mean_lst|        lst_anomaly|consecutive_dry_days|fire_occurred|
+--------+----+------------------+-------------------+--------------------+-------------+
|    ACEH|2019| 31.46368421052631|  5.113636836896337|                  69|            0|
|    ACEH|2019|27.309166666666727| 0.9591192930367534|                  46|            1|
|    ACEH|2019|30.372500000000002|  4.022452626370029|                  47|            0|
|    ACEH|2019|28.922799999999995|  2.572752626370022|                  78|            0|
|    ACEH|2019| 26.66371428571432|0.31366691208434716|                  48|            0|
+--------+----+------------------+-------------------+--------------------+-------------+
only showing top 5 rows

Missing value counts:
+--------+--------------------+-----------------------+------------------+-------------+--------+---------+-------+-------------------+--------+
|province|consecutive_dry_days|days_since_season_start|days_to_4mm_breach|fire_occurred|latitude|longitude|lst_max|mean_daily_rainfall|mean_lst|
+--------+--------------------+-----------------------+------------------+-------------+--------+---------+-------+-------------------+--------+
|       0|                   0|                      0|                 0|            0|       0|        0|      0|                  0|       0|
+--------+--------------------+-----------------------+------------------+-------------+--------+---------+-------+-------------------+--------+


✓ col imported for temporal split operations

### Train/Test Split

**Temporal split (prevents data leakage):**
- Train: 2015-2021 (includes 2015 + 2019 El Niño events)
- Test: 2022-2023 (held-out future seasons)

Temporal split function defined

→ Splitting data: train ≤ 2019, test > 2019
✓ Train: 165 rows (27.1%)
✓ Test:  443 rows (72.9%)

Training set class balance:
  Fire:    5 (3.03%)
  No fire: 160 (96.97%)

Train sample:
+--------+----+--------------------+-------------+
|province|year|consecutive_dry_days|fire_occurred|
+--------+----+--------------------+-------------+
|    ACEH|2019|                  69|            0|
|    ACEH|2019|                  46|            1|
|    ACEH|2019|                  47|            0|
+--------+----+--------------------+-------------+
only showing top 3 rows

Test sample:
+--------+----+--------------------+-------------+
|province|year|consecutive_dry_days|fire_occurred|
+--------+----+--------------------+-------------+
|   JAMBI|2022|                 102|            1|
|   JAMBI|2022|                  93|            0|
|   JAMBI|2022|                  93|            0|
+--------+----+--------------------+-------------+
only showing top 3 rows

### Cache to Google Drive

Save train/test splits as parquet for fast loading in Phase 3 (model training).

Caching function defined

→ Caching splits to /content/drive/MyDrive/forest-fire-data/processed...
✓ Saved train to /content/drive/MyDrive/forest-fire-data/processed/train.parquet
✓ Saved test to /content/drive/MyDrive/forest-fire-data/processed/test.parquet

Verified:
  Train: 165 rows
  Test:  443 rows

✓ Cache test successful
  Files saved to /content/drive/MyDrive/forest-fire-data/processed

### End-to-End Pipeline

Combine all steps: load → engineer → split → cache

Full pipeline function defined

**Uncomment below to run full pipeline after Task 4 completes:**

---

## Phase 1+2 Complete ✓

**Completed:**
- ✓ Environment detection (Colab/local)
- ✓ Earth Engine authentication
- ✓ Google Drive mount
- ✓ Test export (Riau 2019)
- ✓ Full aggregation functions (ready for batch export)
- ✓ PySpark preprocessing pipeline
- ✓ Train/test split (2015-2021 | 2022-2023)
- ✓ Caching to Drive

**Next Steps (Phase 3):**
1. Uncomment full export in Task 4 (creates ~7M row dataset)
2. Run full preprocessing pipeline
3. Train XGBoost Classifier (fire probability)
4. Train XGBoost Regressor (days to 4mm breach)
5. Evaluate on 2022-2023 test set
6. Generate district-level risk CSV

**To run full export:**
```python
# Scroll to Task 4, uncomment:
# all_years_data = batch_export_all_years(target_provinces, years=range(2015, 2024))

# Then scroll to Task 6, uncomment:
# train_path, test_path = run_full_preprocessing_pipeline()
```

**Estimated time for full export:** 1-3 hours (runs in background)

# phase_3_model_training_updated.ipynb

# Phase 3: Model Training - Peatland Fire Prediction

This notebook trains dual XGBoost models for fire prediction:
1. **Classification**: Predicts fire occurrence (binary)
2. **Regression**: Predicts days until 4mm rainfall breach (continuous)

**Data source**: Preprocessed train/test parquet files from Phase 2

Environment: Colab

Mounted at /content/drive
✓ Drive mounted

✓ Libraries imported

Train file exists: True
Test file exists: True

✓ Train: 23,169 rows
✓ Test:  3,310 rows

Train schema:
province                       object
consecutive_dry_days            int64
days_since_season_start         int64
days_to_4mm_breach              int64
fire_occurred                   int32
latitude                      float64
longitude                     float64
lst_max                       float64
mean_daily_rainfall           float64
mean_lst                      float64
min_ndvi                      float64
province_id                    object
season                         object
season_year                     int64
total_rainfall_season         float64
year                            int64
province_baseline_lst         float64
province_baseline_rainfall    float64
lst_anomaly                   float64
rainfall_deficit_30d          float64
province_index                float64
province_vector                object
dtype: object

✓ Exploded province_vector into 9 columns: ['prov_0', 'prov_1', 'prov_2']...
Train shape: (23169, 30)
Test shape:  (3310, 30)

Base features: 9
Province features: 9
Total features: 18
Targets: fire_occurred, days_to_4mm_breach

X_train shape: (23169, 18)
X_test shape:  (3310, 18)

Class balance (train):
fire_occurred
0    0.948897
1    0.051103
Name: proportion, dtype: float64

Grid search combinations: 16 = 16

✓ Pipeline created: SMOTE → XGBClassifier

NaN counts per feature:
consecutive_dry_days       0
days_since_season_start    0
lst_max                    0
mean_daily_rainfall        0
mean_lst                   0
min_ndvi                   0
total_rainfall_season      0
lst_anomaly                0
rainfall_deficit_30d       0
prov_0                     0
prov_1                     0
prov_2                     0
prov_3                     0
prov_4                     0
prov_5                     0
prov_6                     0
prov_7                     0
prov_8                     0
dtype: int64

Total NaN rows: 0

Starting grid search (this may take 10-20 minutes)...
SMOTE will be applied inside each CV fold (no data leakage)
Fitting 5 folds for each of 16 candidates, totalling 80 fits


✓ Best F1 Score: 0.1827
✓ Best params: {'clf__colsample_bytree': 0.8, 'clf__learning_rate': 0.1, 'clf__max_depth': 6, 'clf__min_child_weight': 1, 'clf__n_estimators': 200, 'clf__scale_pos_weight': 3, 'clf__subsample': 0.8}

✓ Fire classifier trained
Best hyperparameters: {'clf__colsample_bytree': 0.8, 'clf__learning_rate': 0.1, 'clf__max_depth': 6, 'clf__min_child_weight': 1, 'clf__n_estimators': 200, 'clf__scale_pos_weight': 3, 'clf__subsample': 0.8}

✓ Predictions generated: 3310 samples
Predicted fire cases: 1080 (32.63%)

============================================================
FIRE CLASSIFIER EVALUATION
============================================================
ROC-AUC: 0.6089

Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.68      0.80      3253
           1       0.02      0.46      0.05        57

    accuracy                           0.67      3310
   macro avg       0.51      0.57      0.42      3310
weighted avg       0.97      0.67      0.79      3310


Fire cases in training: 1,184 (5.11%)
Days-to-breach range: [13, 58]
Mean: 43.2 days

Grid search combinations: 216 = 216

✓ Base regressor initialized

Starting regressor grid search (this may take 15-25 minutes)...
Fitting 5 folds for each of 216 candidates, totalling 1080 fits

✓ Best MAE: 0.01 days
✓ Best params: {'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 1, 'n_estimators': 300, 'subsample': 1.0}

✓ Days-to-breach regressor trained
Best hyperparameters: {'objective': 'reg:squarederror', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': 1.0, 'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': None, 'feature_types': None, 'gamma': None, 'gpu_id': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.05, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 8, 'max_leaves': None, 'min_child_weight': 1, 'missing': nan, 'monotone_constraints': None, 'n_estimators': 300, 'n_jobs': None, 'num_parallel_tree': None, 'predictor': None, 'random_state': 42, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': None, 'subsample': 1.0, 'tree_method': 'hist', 'validate_parameters': None, 'verbosity': None}

✓ Predictions for 57 fire cases
Predicted days range: [18.0, 33.0]
Actual days range: [18, 33]

============================================================
DAYS-TO-BREACH REGRESSOR EVALUATION
============================================================
MAE:  0.02 days
RMSE: 0.13 days
R²:   0.9989

✓ Libraries tambahan (Random Forest & LightGBM) di-import

Starting Random Forest Classifier grid search...
Fitting 5 folds for each of 16 candidates, totalling 80 fits

✓ Best F1 Score (RF): 0.2502
✓ Best params: {'clf__class_weight': 'balanced_subsample', 'clf__max_depth': 10, 'clf__min_samples_split': 5, 'clf__n_estimators': 200}
============================================================
RANDOM FOREST CLASSIFIER EVALUATION
============================================================
ROC-AUC: 0.6313
              precision    recall  f1-score   support

           0       0.99      0.72      0.83      3253
           1       0.03      0.42      0.05        57

    accuracy                           0.71      3310
   macro avg       0.51      0.57      0.44      3310
weighted avg       0.97      0.71      0.82      3310


Starting Random Forest Regressor grid search...
Fitting 5 folds for each of 12 candidates, totalling 60 fits

✓ Best MAE (RF): 0.02 days
✓ Best params: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
============================================================
RANDOM FOREST REGRESSOR EVALUATION
============================================================
MAE:  0.04 days
RMSE: 0.15 days
R²:   0.9987

Starting LightGBM Classifier grid search...
Fitting 5 folds for each of 8 candidates, totalling 40 fits

✓ Best F1 Score (LGBM): 0.2413
✓ Best params: {'clf__is_unbalance': True, 'clf__learning_rate': 0.05, 'clf__max_depth': 4, 'clf__n_estimators': 100}
============================================================
LIGHTGBM CLASSIFIER EVALUATION
============================================================
ROC-AUC: 0.6286
              precision    recall  f1-score   support

           0       0.99      0.65      0.78      3253
           1       0.02      0.47      0.04        57

    accuracy                           0.64      3310
   macro avg       0.50      0.56      0.41      3310
weighted avg       0.97      0.64      0.77      3310


Starting LightGBM Regressor grid search...
Fitting 5 folds for each of 8 candidates, totalling 40 fits

✓ Best MAE (LGBM): 0.11 days
✓ Best params: {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 200}
============================================================
LIGHTGBM REGRESSOR EVALUATION
============================================================
MAE:  0.50 days
RMSE: 0.91 days
R²:   0.9487

PERBANDINGAN CLASSIFIER
        Model  roc_auc  precision   recall       f1  accuracy
      XGBoost 0.608933   0.024074 0.456140 0.045734  0.672205
Random Forest 0.631347   0.025614 0.421053 0.048290  0.714199
     LightGBM 0.628564   0.023018 0.473684 0.043902  0.644713

PERBANDINGAN REGRESSOR
        Model      mae     rmse       r2
      XGBoost 0.017589 0.132438 0.998904
Random Forest 0.044386 0.146919 0.998652
     LightGBM 0.500834 0.906026 0.948726
✓ Saved comparison plot: /content/drive/MyDrive/forest-fire-data/models/model_comparison.png

✓ Grafik Regressor tersimpan di: /content/drive/MyDrive/forest-fire-data/models/model_comparison_regressor.png

============================================================
✓ SEMUA MODEL BERHASIL DILATIH DAN DISIMPAN
============================================================

✓ Gambar Confusion Matrix tersimpan di: /content/drive/MyDrive/forest-fire-data/models/model_comparison_confusion_matrix.png


✓ Gambar Scatter Plot tersimpan di: /content/drive/MyDrive/forest-fire-data/models/model_comparison_scatter.png


✓ Saved: /content/drive/MyDrive/forest-fire-data/models/feature_importance.png


Top 5 features:
                 feature  importance
2                lst_max    0.205514
4               mean_lst    0.195353
8   rainfall_deficit_30d    0.153547
3    mean_daily_rainfall    0.098545
6  total_rainfall_season    0.097249

✓ Saved: /content/drive/MyDrive/forest-fire-data/models/fire_classifier.joblib
  Size: 0.71 MB

✓ Saved: /content/drive/MyDrive/forest-fire-data/models/days_regressor.joblib
  Size: 0.33 MB

✓ Metrics aggregated with placeholder target warning

✓ Saved: /content/drive/MyDrive/forest-fire-data/models/training_metrics.json

✓ Classifier loads correctly (test prediction: [0 0 0 1 0])
✓ Regressor loads correctly (test prediction: [25.000025 30.000002 33.999996 39.99999  37.99999 ])

============================================================
PHASE 3: MODEL TRAINING COMPLETE
============================================================

PERBANDINGAN CLASSIFIER
        Model  roc_auc  precision   recall       f1  accuracy
      XGBoost 0.608933   0.024074 0.456140 0.045734  0.672205
Random Forest 0.631347   0.025614 0.421053 0.048290  0.714199
     LightGBM 0.628564   0.023018 0.473684 0.043902  0.644713

PERBANDINGAN REGRESSOR
        Model      mae     rmse       r2
      XGBoost 0.017589 0.132438 0.998904
Random Forest 0.044386 0.146919 0.998652
     LightGBM 0.500834 0.906026 0.948726

✓ Ready for Phase 4: District Risk Assessment (All Models)

# Phase 4: District Risk Assessment (All Models)
Menghitung skor risiko distrik menggunakan masing-masing model (XGBoost, Random Forest, LightGBM).


================ XGBoost ================
✓ Aggregated to 10 provinces
✓ Risk scores calculated
✓ Saved: /content/drive/MyDrive/forest-fire-data/models/district_risk_assessment_xgboost.csv

Top 5 highest risk provinces (XGBoost):
            province  risk_score risk_category
0   SUMATERA SELATAN   99.999989      Critical
1   KALIMANTAN BARAT   94.131428      Critical
2               RIAU   89.355501      Critical
3              JAMBI   87.609950      Critical
4  KALIMANTAN TENGAH   60.655817          High

------------------------------------------------------------

================ RandomForest ================
✓ Aggregated to 10 provinces
✓ Risk scores calculated
✓ Saved: /content/drive/MyDrive/forest-fire-data/models/district_risk_assessment_randomforest.csv

Top 5 highest risk provinces (RandomForest):
            province  risk_score risk_category
0   SUMATERA SELATAN   99.999990      Critical
1              JAMBI   78.277671      Critical
2               RIAU   73.076151          High
3   KALIMANTAN BARAT   68.370457          High
4  KALIMANTAN TENGAH   45.869913      Moderate

------------------------------------------------------------

================ LightGBM ================
✓ Aggregated to 10 provinces
✓ Risk scores calculated
✓ Saved: /content/drive/MyDrive/forest-fire-data/models/district_risk_assessment_lightgbm.csv

Top 5 highest risk provinces (LightGBM):
           province  risk_score risk_category
0  SUMATERA SELATAN   99.999990      Critical
1              RIAU   85.193234      Critical
2             JAMBI   84.473667      Critical
3  KALIMANTAN BARAT   72.630495          High
4    SUMATERA BARAT   55.428574          High

------------------------------------------------------------

