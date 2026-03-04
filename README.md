
# ⚓ MaritimePort AI

MaritimePort AI is an end-to-end **AI-powered maritime port intelligence system** designed to support
**operational decision-making at container ports** using large-scale AIS data, port operations data,
and machine learning models.

The system focuses on three core predictive capabilities:
1. **Berth Feasibility Prediction**
2. **Port Congestion Classification**
3. **Delay Estimation**

The project follows a **Medallion Architecture (Raw → Bronze → Silver → Gold)** and is delivered
through a **Streamlit-based decision-support web application**, fully Dockerized for deployment.

---

## 🚀 Key Features

- Large-scale AIS data processing (tens of millions of records)
- Multi-layer data engineering (Raw, Bronze, Silver, Gold)
- ML-driven operational intelligence
- Policy-aware AI decision agent (RAG over port tariff PDF)
- Dockerized deployment
- Interactive Streamlit UI

---

## 📊 Dataset Sizes Across the Pipeline

MaritimePort AI processes raw AIS data through progressive refinement stages.
The table below reflects **actual dataset sizes from the final pipeline**.

### 🔹 Raw Layer (AIS Ingestion)

- **Files:** 365 daily AIS CSV files (Jan–Dec 2024)
- **Records per file:** ~80,000 – 120,000
- **Total records:** ~4–5 million AIS points
- **Columns:** ~18
- **Purpose:** Unprocessed, schema-on-read AIS data

---

### 🔹 Bronze Layer (LA Port & Anchorage Filtered AIS)

- **Filters applied:**
  - Cargo vessels only (VesselType 70–79)
  - LA Port & Anchorage geofence
  - Speed-based location logic
- **Output files:** 365
- **Rows per file:** ~8,500 – 12,000
- **Total rows (combined):** **4,177,218**
- **Columns:** 18
- **Key output:** `final_combined_df.csv`

---

### 🔹 Silver Layer (Integrated Operational Dataset)

- **Data sources:**
  - Aggregated AIS behavior
  - Port calls (arrival, berth, departure)
  - Terminal & berth operations
- **Output file:** `la_integrated_dataset_2024.csv`
- **Rows:** **233,520**
- **Columns:** **36**
- **Granularity:** Port call / vessel level

---

### 🔹 Gold Layer (ML-Ready Feature Store)

- **Output file:** `la_port_ml_ready_dataset.csv`
- **Rows:** **233,520**
- **Columns:** **14**
- **Targets:**
  - `BerthFeasible`
  - `CongestionLevel`
  - `DelayMinutes`
- **Purpose:** Final training & inference dataset

---

### 📉 Dataset Reduction Summary

| Layer | Rows |
|------|------|
| Raw | ~4–5 million |
| Bronze | 4,177,218 |
| Silver | 233,520 |
| Gold | 233,520 |

➡️ ~95% data reduction with **no loss of decision-critical information**.

---

## 🧠 Final Models Used

| Use Case | Model | File |
|--------|------|------|
| Berth Feasibility | XGBoost Classifier | `xgb_berth_feasibility.pkl` |
| Port Congestion | Random Forest Classifier | `rf_congestion.pkl` |
| Delay Minutes | LightGBM Regressor | `lgbm_regression_model.pkl` |

---

## 📂 Project Structure

```
MaritimePort_AI/
│── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│── notebooks/
│── models/
│── Port_AI_WebApp/
│── Dockerfile
│── policy.pdf
│── README.md
```

---

## 🔄 End-to-End Data Flow

1. Raw AIS ingestion
2. Geospatial filtering
3. Bronze aggregation
4. Silver integration
5. Gold feature engineering
6. Model training & selection
7. Real-time inference via Streamlit
8. Policy-aware decision explanation (RAG)

---

## 🌐 Web Application (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🐳 Docker Deployment

```bash
docker build -t maritimeport-ai .
docker run -p 8501:8501 maritimeport-ai
```

---

## 📌 Use Cases

- Port authority decision support
- Berth allocation planning
- Congestion risk monitoring
- Operational delay mitigation
- AI-assisted policy interpretation

---

## ⚠️ Disclaimer

Academic and decision-support use only.

---

## 👨‍💻 Team

* **Harshit Arora**
* **Yash Daund**
* **Mayuresh Bhombe**
* **Aashi Chahal**
* **Prapti Kinare**
* **Pravin Garje**
- C-DAC Major Project – AI & Data Engineering
