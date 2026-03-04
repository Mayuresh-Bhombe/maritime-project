
# 🏗️ MaritimePort AI – System Architecture

## 1. High-Level Architecture

```
AIS CSVs
   |
   v
[ RAW LAYER ]
   |
   v
[ BRONZE LAYER ]
(LA Port & Anchorage Filtering)
   |
   v
[ SILVER LAYER ]
(AIS + Port Calls + Ops Integration)
   |
   v
[ GOLD LAYER ]
(ML-Ready Feature Store)
   |
   v
[ ML MODELS ]
   |
   v
[ STREAMLIT APP ]
   |
   v
[ AI DECISION AGENT + RAG ]
```

---

## 2. Data Architecture (Medallion Pattern)

### 🔹 Raw
- Unprocessed AIS daily dumps
- High volume, schema-on-read

### 🔹 Bronze
- Geospatially filtered AIS data
- Cargo vessels only
- Port vs anchorage classification

### 🔹 Silver
- Vessel-level aggregation
- Integrated with:
  - Port calls
  - Terminal capacity
  - Berth metadata

### 🔹 Gold
- Feature-engineered ML datasets
- Targets:
  - `BerthFeasible`
  - `CongestionLevel`
  - `DelayMinutes`

---

## 3. ML Architecture

| Layer | Responsibility |
|-----|---------------|
| Feature Store | Gold CSV |
| Training | Jupyter Notebooks |
| Model Registry | `.pkl` files |
| Inference | Streamlit runtime |

---

## 4. AI Decision Agent (RAG)

- PDF: Port of LA Tariff
- Chunking with overlap
- TF-IDF vectorization
- Cosine similarity retrieval
- LLM-generated explanations (fallback safe mode)

---

## 5. Deployment Architecture

- Dockerized Python app
- Streamlit frontend
- Stateless inference
- Local or cloud deployable

---

## 6. Design Principles

- Scalability-first data design
- Model explainability
- Clear separation of concerns
- Production-oriented structure

---

## 7. Future Enhancements

- Live AIS streaming
- Time-series forecasting
- Multi-port generalization
- MLOps (model versioning & monitoring)
