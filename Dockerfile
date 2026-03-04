FROM python:3.10-slim

# ---- System dependency for LightGBM / XGBoost ----
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ----
COPY Port_AI_WebApp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application files ----
COPY Port_AI_WebApp ./Port_AI_WebApp
COPY models ./models
COPY policy.pdf ./policy.pdf
COPY Port_AI_WebApp/.streamlit ./Port_AI_WebApp/.streamlit

EXPOSE 8501

# ---- Run Streamlit ----
CMD ["streamlit", "run", "Port_AI_WebApp/app.py", "--server.address=0.0.0.0"]
