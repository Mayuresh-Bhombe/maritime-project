# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="MaritimePort AI",
    page_icon="⚓",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================
BASE_PATH = Path(__file__).resolve().parents[1]
MODELS_PATH = BASE_PATH / "models"
POLICY_PATH = BASE_PATH / "policy.pdf"

# =====================================================
# LOAD MODELS
# =====================================================
rf_congestion = joblib.load(MODELS_PATH / "rf_congestion.pkl")
lgb_delay = joblib.load(MODELS_PATH / "lgbm_regression_model.pkl")
xgb_berth = joblib.load(MODELS_PATH / "xgb_berth_feasibility.pkl")

CONGESTION_LABELS = {
    0: "Low",
    1: "High"
}

TERMINAL_MAP = {f"T{i}": i for i in range(1, 8)}
BERTH_MAP = {f"B{i}": i for i in range(1, 11)}

# =====================================================
# GEMINI SAFE CONFIG
# =====================================================
import os
llm = None
try:
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel("models/gemini-2.5-flash")
except Exception:
    llm = None


def safe_llm(prompt: str) -> str:
    if llm is None:
        return """
### ⚠️ AI Explanation Unavailable

Gemini API key missing or invalid.
ML predictions are still valid.
"""
    try:
        return llm.generate_content(prompt).text
    except Exception as e:
        return f"""
### ⚠️ AI Processing Failed
{str(e)}
"""

# =====================================================
# POLICY RAG
# =====================================================
@st.cache_resource
def load_policy():
    reader = PdfReader(str(POLICY_PATH))
    text = "".join(p.extract_text() for p in reader.pages)

    chunks, size, overlap = [], 800, 100
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunks)
    return chunks, vectorizer, X

policy_chunks, policy_vectorizer, policy_X = load_policy()

def get_policy_context(query, k=2):
    q_vec = policy_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, policy_X)[0]
    idx = np.argsort(sims)[-k:][::-1]
    return "\n".join(policy_chunks[i] for i in idx)

def decision_agent(title, scenario, prediction):
    policy = get_policy_context(title)
    prompt = f"""
You are a maritime port decision-support AI.

Note:
Congestion is classified as BINARY:
- Low congestion
- High congestion

Scenario:
{scenario}

Prediction:
{prediction}

Port Policy:
{policy}

MANDATORY FORMAT:

Cause:
- bullet
- bullet

Recommended Actions:
- action
- action
"""
    return safe_llm(prompt)

# =====================================================
# UI
# =====================================================
st.title("⚓ MaritimePort AI")
st.caption("AI-powered decision intelligence for maritime port operations")


tabs = st.tabs([
    "🚦 Congestion",
    "⏱️ Delay Minutes",
    "⚓ Berth Feasibility",
    "🧠 Combined Decision",
    "💬 Policy Chat"
])

# =====================================================
# TAB 1 — CONGESTION
# =====================================================
with tabs[0]:
    ArrivalHour = st.slider("Arrival Hour", 0, 23, 10, key="c_hr")
    ArrivalDay = st.selectbox("Arrival Day", range(7), key="c_day")
    BerthTime = st.number_input("Berth Time (hrs)", 1, 120, 24, key="c_bt")
    Capacity = st.number_input("Daily Capacity (TEU)", 500, 20000, 9000, key="c_cap")
    Feasible = st.selectbox("Berth Feasible", [0, 1], key="c_feas")

    df = pd.DataFrame([{
        "BerthTime": BerthTime,
        "ArrivalHour": ArrivalHour,
        "BerthFeasible": Feasible,
        "DailyCapacityTEU": Capacity,
        "ArrivalDayOfWeek": ArrivalDay
    }])

    scenario = st.text_area(
        "Scenario Prompt (Editable)",
        f"""
Arrival Hour: {ArrivalHour}
Day: {ArrivalDay}
Berth Time: {BerthTime}
Daily Capacity: {Capacity}
Berth Feasible: {Feasible}
""",
        key="c_prompt"
    )

    if st.button("Run Congestion Prediction"):
        label = CONGESTION_LABELS[rf_congestion.predict(df)[0]]
        st.success(f"Congestion Level: {label}")

        with st.spinner("🧠 Analyzing congestion using port policy & AI..."):
            output = decision_agent("Port Congestion", scenario, label)

        st.markdown("### 🧠 AI Decision Analysis")
        st.markdown(output)

# =====================================================
# TAB 2 — DELAY
# =====================================================
with tabs[1]:
    ArrivalHour = st.slider("Arrival Hour", 0, 23, 10, key="d_hr")
    ArrivalDay = st.selectbox("Arrival Day", range(7), key="d_day")
    BerthTime = st.number_input("Berth Time (hrs)", 1, 120, 24, key="d_bt")
    Feasible = st.selectbox("Berth Feasible", [0, 1], key="d_feas")

    df = pd.DataFrame([{
        "ArrivalHour": ArrivalHour,
        "ArrivalDayOfWeek": ArrivalDay,
        "BerthFeasible": Feasible,
        "BerthTime": BerthTime
    }])

    scenario = st.text_area(
        "Scenario Prompt (Editable)",
        f"""
Arrival Hour: {ArrivalHour}
Day: {ArrivalDay}
Berth Time: {BerthTime}
Berth Feasible: {Feasible}
""",
        key="d_prompt"
    )

    if st.button("Run Delay Prediction"):
        delay = round(float(lgb_delay.predict(df)[0]), 2)
        st.metric("Delay (minutes)", delay)

        with st.spinner("🧠 Evaluating delay impact using AI & policy..."):
            output = decision_agent("Delay Analysis", scenario, delay)

        st.markdown("### 🧠 AI Decision Analysis")
        st.markdown(output)

# =====================================================
# TAB 3 — BERTH FEASIBILITY
# =====================================================
with tabs[2]:
    Draft = st.number_input("Vessel Draft (m)", 5.0, 20.0, 10.0, key="b_draft")
    ArrivalHour = st.slider("Arrival Hour", 0, 23, 10, key="b_hr")
    ArrivalDay = st.selectbox("Arrival Day", range(7), key="b_day")
    Terminal = st.selectbox("Terminal", list(TERMINAL_MAP.keys()), key="b_term")
    Berth = st.selectbox("Berth", list(BERTH_MAP.keys()), key="b_berth")

    df = pd.DataFrame([{
        "VesselDraft": Draft,
        "ArrivalHour": ArrivalHour,
        "ArrivalDayOfWeek": ArrivalDay,
        "TerminalID": TERMINAL_MAP[Terminal],
        "BerthID": BERTH_MAP[Berth]
    }])

    scenario = st.text_area(
        "Scenario Prompt (Editable)",
        f"""
Vessel Draft: {Draft}
Arrival Hour: {ArrivalHour}
Terminal: {Terminal}
Berth: {Berth}
""",
        key="b_prompt"
    )

    if st.button("Run Feasibility Check"):
        feasible = "Yes" if xgb_berth.predict(df)[0] == 1 else "No"
        st.success(f"Berth Feasible: {feasible}")

        with st.spinner("🧠 Validating berth feasibility using AI & policy..."):
            output = decision_agent("Berth Feasibility", scenario, feasible)

        st.markdown("### 🧠 AI Decision Analysis")
        st.markdown(output)

# =====================================================
# TAB 4 — COMBINED
# =====================================================
with tabs[3]:
    st.info("Integrated system-level decision across berth feasibility, congestion, and delay.")

    ArrivalHour = st.slider("Arrival Hour", 0, 23, 10, key="x_hr")
    ArrivalDay = st.selectbox("Arrival Day", range(7), key="x_day")
    BerthTime = st.number_input("Berth Time (hrs)", 1, 120, 24, key="x_bt")
    Capacity = st.number_input("Daily Capacity (TEU)", 500, 20000, 9000, key="x_cap")
    Draft = st.number_input("Vessel Draft (m)", 5.0, 20.0, 10.0, key="x_draft")
    Terminal = st.selectbox("Terminal", list(TERMINAL_MAP.keys()), key="x_term")
    Berth = st.selectbox("Berth", list(BERTH_MAP.keys()), key="x_berth")

    if st.button("Run Combined Decision"):
        berth_df = pd.DataFrame([{
            "VesselDraft": Draft,
            "ArrivalHour": ArrivalHour,
            "ArrivalDayOfWeek": ArrivalDay,
            "TerminalID": TERMINAL_MAP[Terminal],
            "BerthID": BERTH_MAP[Berth]
        }])

        berth_feasible = xgb_berth.predict(berth_df)[0]

        cong_df = pd.DataFrame([{
            "BerthTime": BerthTime,
            "ArrivalHour": ArrivalHour,
            "BerthFeasible": berth_feasible,
            "DailyCapacityTEU": Capacity,
            "ArrivalDayOfWeek": ArrivalDay
        }])

        congestion = CONGESTION_LABELS[rf_congestion.predict(cong_df)[0]]

        delay_df = pd.DataFrame([{
            "ArrivalHour": ArrivalHour,
            "ArrivalDayOfWeek": ArrivalDay,
            "BerthFeasible": berth_feasible,
            "BerthTime": BerthTime
        }])

        delay = round(float(lgb_delay.predict(delay_df)[0]), 2)

        st.metric("Berth Feasible", "Yes" if berth_feasible else "No")
        st.metric("Congestion", congestion)
        st.metric("Delay (minutes)", delay)

        scenario = f"""
Arrival Hour: {ArrivalHour}
Day: {ArrivalDay}
Berth Time: {BerthTime}
Capacity: {Capacity}
Draft: {Draft}
Terminal: {Terminal}
Berth: {Berth}
"""

        with st.spinner("🧠 Performing integrated port decision analysis..."):
            output = decision_agent(
                "Combined Port Decision",
                scenario,
                f"Feasible={berth_feasible}, Congestion={congestion}, Delay={delay}"
            )

        st.markdown("### 🧠 AI Decision Analysis")
        st.markdown(output)

# =====================================================
# TAB 5 — POLICY CHAT
# =====================================================
with tabs[4]:
    q = st.text_area("Ask policy-based question")
    if st.button("Ask AI"):
        with st.spinner("🧠 Searching port policy..."):
            out = decision_agent("Port Policy Query", q, "N/A")
        st.markdown("### 🧠 Policy-Based Answer")
        st.markdown(out)
