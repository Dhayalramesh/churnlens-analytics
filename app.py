"""ChurnLens: Streamlit dashboard (overview, model, at-risk customers, ask-the-data, pipeline)."""
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from churnlens import __version__
from churnlens.algorithms import TopK
from churnlens.bootstrap import ensure_artifacts
from churnlens.config import DB_PATH, EXPORT_PATH, METRICS_PATH, RAW_CSV
from churnlens.llm import MAX_QUESTION_CHARS, InsightAssistant, UnsafeSQL, make_client, pick_model
from churnlens.modeling import FEATURES, ChurnModeler
from churnlens.pipeline import DataValidator
from churnlens.sql import QUERIES, SQLRepository

REPO_URL = "https://github.com/Dhayalramesh/churnlens-analytics"
SESSION_LLM_LIMIT = 10
TEAL, BLUE, AMBER = "#2dd4bf", "#4cb5ff", "#f5a524"

st.set_page_config(page_title="ChurnLens", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner="First start: preparing the data and training the models (about 1-2 minutes)...")
def bootstrap() -> list[str]:
    return ensure_artifacts()


bootstrap()


def _secret(name: str):
    try:
        value = st.secrets[name]
    except Exception:
        value = None
    return value or os.environ.get(name)


@st.cache_data(show_spinner=False)
def run_named(name: str) -> pd.DataFrame:
    return SQLRepository(DB_PATH).named(name)


@st.cache_data(show_spinner=False)
def run_sql(sql: str) -> pd.DataFrame:
    return SQLRepository(DB_PATH).query(sql)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(Path(METRICS_PATH).read_text())


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return ChurnModeler().load_model()


@st.cache_data(show_spinner=False)
def customers_frame() -> pd.DataFrame:
    return run_sql("SELECT * FROM customers")


@st.cache_data(show_spinner=False)
def current_with_risk() -> pd.DataFrame:
    return run_sql("""SELECT c.customerID, c.Contract, c.InternetService, c.PaymentMethod, c.tenure, c.MonthlyCharges,
                             p.churn_probability
                      FROM customers c JOIN predictions p ON p.customerID = c.customerID WHERE c.churn_flag = 0""")


@st.cache_resource(show_spinner=False, ttl=900)
def build_assistant(key):
    if not key:
        return None, "No GROQ_API_KEY secret found: only the ready-made queries are available."
    try:
        client = make_client(key)
        model = pick_model(client)
        if not model:
            return None, "No supported model is available for this key."
        return InsightAssistant(client, model), None
    except Exception as exc:
        return None, f"LLM unavailable ({type(exc).__name__}). Ready-made queries still work."


def bar(df: pd.DataFrame, x: str, y: str, color: str, title: str, sort=None, horizontal=False):
    enc_x, enc_y = (alt.X(y, title=None), alt.Y(x, sort=sort, title=None)) if horizontal else (alt.X(x, sort=sort, title=None), alt.Y(y, title=None))
    return alt.Chart(df, title=title).mark_bar(color=color).encode(x=enc_x, y=enc_y, tooltip=list(df.columns)).properties(height=260)


metrics = load_metrics()
best = metrics["best_model"]
tuned = metrics["test_at_tuned_threshold"]

with st.sidebar:
    st.markdown("### ChurnLens")
    st.caption(f"v{__version__} | customer-churn analytics")
    st.write(f"**Best model:** {best}")
    st.write(f"**Test ROC-AUC:** {tuned['roc_auc']}")
    st.caption("Data: IBM Telco customer-churn sample (7,043 customers, public).")
    st.markdown(f"[Source on GitHub]({REPO_URL})")

st.title("ChurnLens")
st.caption("ETL pipeline, SQL analytics, machine learning and an optional LLM question-answering layer on customer-churn data.")
tab_over, tab_model, tab_cust, tab_ask, tab_pipe = st.tabs(["Overview", "Model", "At-risk customers", "Ask the data", "Data pipeline"])

# ------------------------------------------------------------------ overview
with tab_over:
    k = run_named("kpis").iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{int(k.customers):,}")
    b.metric("Churn rate", f"{k.churn_rate_pct}%")
    c.metric("Avg monthly charge", f"${k.avg_monthly_charge}")
    d.metric("Monthly revenue lost to churn", f"${k.monthly_revenue_lost:,.0f}")
    left, right = st.columns(2)
    with left:
        st.altair_chart(bar(run_named("churn_by_contract"), "Contract", "churn_rate_pct", TEAL, "Churn rate by contract (%)", sort="-y"), width="stretch")
    with right:
        tenure = run_named("churn_by_tenure")
        st.altair_chart(bar(tenure, "tenure_bucket", "churn_rate_pct", BLUE, "Churn rate by tenure (%)", sort=list(tenure.tenure_bucket)), width="stretch")
    st.markdown("**Riskiest customer segments** (SQL window function: top 3 per contract)")
    st.dataframe(run_named("riskiest_segments"), width="stretch", hide_index=True)
    st.markdown("**Customers paying above their contract's average price** (window function)")
    st.dataframe(run_named("above_contract_average_price"), width="stretch", hide_index=True)

# ------------------------------------------------------------------ model
with tab_model:
    st.markdown("#### Model comparison (5-fold cross-validation on the training split)")
    st.dataframe(pd.DataFrame(metrics["comparison"]), width="stretch", hide_index=True)
    st.caption(f"Best: {best}. The decision threshold ({metrics['tuned_threshold']}) was tuned on out-of-fold predictions, not on the test set.")
    a, b, c, d = st.columns(4)
    a.metric("Test ROC-AUC", tuned["roc_auc"])
    b.metric("Recall (churners caught)", tuned["recall"])
    c.metric("Precision", tuned["precision"])
    d.metric("F1", tuned["f1"])
    left, right = st.columns(2)
    with left:
        roc = pd.DataFrame(metrics["roc_curve"])
        line = alt.Chart(roc, title="ROC curve (test set)").mark_line(color=TEAL).encode(x=alt.X("fpr", title="False positive rate"), y=alt.Y("tpr", title="True positive rate"))
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="grey", strokeDash=[4, 4]).encode(x="x", y="y")
        st.altair_chart((line + diag).properties(height=280), width="stretch")
    with right:
        imp = pd.DataFrame(metrics["feature_importance"])
        st.altair_chart(bar(imp, "feature", "importance", AMBER, "Permutation importance (drop in ROC-AUC)", sort="-x", horizontal=True), width="stretch")
    cm = tuned["confusion_matrix"]
    st.markdown("**Confusion matrix (test set)**")
    st.dataframe(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], index=["Actually stayed", "Actually churned"],
                              columns=["Predicted stay", "Predicted churn"]), width="stretch")

# ------------------------------------------------------------------ customers
with tab_cust:
    risk = current_with_risk()
    contract = st.multiselect("Contract", sorted(risk.Contract.unique()), default=list(sorted(risk.Contract.unique())))
    top_n = st.slider("Show top N customers by expected monthly revenue at risk", 5, 50, 10)
    view = risk[risk.Contract.isin(contract)]
    heap = TopK(top_n).extend((row.churn_probability * row.MonthlyCharges, row) for row in view.itertuples(index=False))
    table = pd.DataFrame([{**row._asdict(), "expected_monthly_loss": round(score, 2)} for score, row in heap.items()])
    st.caption("Ranked with a bounded min-heap (O(n log k)), not a full sort. Probabilities are out-of-fold, so no customer is scored by a model that saw them.")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download this list (CSV)", table.to_csv(index=False), "at_risk_customers.csv", "text/csv")
    st.altair_chart(bar(run_named("revenue_at_risk_by_contract"), "Contract", "expected_monthly_revenue_at_risk", BLUE,
                        "Expected monthly revenue at risk by contract ($)", sort="-y"), width="stretch")

    st.markdown("#### Score a customer")
    frame = customers_frame()
    c1, c2, c3 = st.columns(3)
    contract_in = c1.selectbox("Contract", sorted(frame.Contract.unique()))
    internet_in = c1.selectbox("Internet service", sorted(frame.InternetService.unique()))
    pay_in = c2.selectbox("Payment method", sorted(frame.PaymentMethod.unique()))
    tech_in = c2.selectbox("Tech support", sorted(frame.TechSupport.unique()))
    tenure_in = c3.slider("Tenure (months)", 0, 72, 12)
    monthly_in = c3.slider("Monthly charge ($)", 18, 120, 70)
    if st.button("Predict churn risk", type="primary"):
        row = {col: (frame[col].median() if pd.api.types.is_numeric_dtype(frame[col]) else frame[col].mode().iloc[0]) for col in FEATURES}
        row.update({"Contract": contract_in, "InternetService": internet_in, "PaymentMethod": pay_in, "TechSupport": tech_in,
                    "tenure": tenure_in, "MonthlyCharges": float(monthly_in), "TotalCharges": float(tenure_in * monthly_in)})
        row["tenure_bucket"] = ("0-12 months" if tenure_in <= 12 else "13-24 months" if tenure_in <= 24 else "25-48 months" if tenure_in <= 48 else "49-72 months")
        proba = float(load_model().predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
        st.metric("Churn probability", f"{proba:.0%}")
        st.write("**High risk**" if proba >= metrics["tuned_threshold"] else "**Lower risk**")
        st.caption("Other fields use the most common value in the data. This is a demonstration, not advice.")

# ------------------------------------------------------------------ ask the data
with tab_ask:
    st.write("Ask a question about the data. Every query is read-only and checked before it runs.")
    st.markdown("**Ready-made queries** (work without any API key)")
    pick = st.selectbox("Query", list(QUERIES), format_func=lambda n: n.replace("_", " "))
    with st.expander("Show SQL"):
        st.code(QUERIES[pick].strip(), language="sql")
    st.dataframe(run_named(pick), width="stretch", hide_index=True)

    st.markdown("**Ask in plain English** (optional LLM, OpenAI-compatible API)")
    assistant, note = build_assistant(_secret("GROQ_API_KEY"))
    if note:
        st.info(note)
    if assistant:
        used = st.session_state.get("llm_questions", 0)
        st.caption(f"Model: {assistant.model} | questions this session: {used}/{SESSION_LLM_LIMIT}")
        question = st.text_input("Your question", placeholder="Which contract type loses the most monthly revenue?", max_chars=MAX_QUESTION_CHARS)
        if st.button("Ask", type="primary", disabled=used >= SESSION_LLM_LIMIT) and question.strip():
            st.session_state["llm_questions"] = used + 1
            try:
                sql = assistant.to_sql(question)
                st.code(sql, language="sql")
                result = run_sql(sql)
                st.dataframe(result, width="stretch", hide_index=True)
                st.write(assistant.explain(question, result.head(20).to_csv(index=False)))
            except UnsafeSQL as exc:
                st.error(f"Blocked: {exc}.")
            except Exception as exc:
                st.error(f"Could not answer that ({type(exc).__name__}). Try rephrasing.")

# ------------------------------------------------------------------ pipeline
with tab_pipe:
    st.markdown("#### ETL stages: extract, validate, transform, validate, load")
    validator = DataValidator()
    raw_report = validator.validate_raw(pd.read_csv(RAW_CSV)).to_frame()
    clean_report = validator.validate_clean(customers_frame()).to_frame()
    st.markdown("**Raw data checks**")
    st.dataframe(raw_report, width="stretch", hide_index=True)
    st.markdown("**Clean data checks**")
    st.dataframe(clean_report, width="stretch", hide_index=True)
    st.markdown("#### Export for Power BI / Tableau / Excel")
    if Path(EXPORT_PATH).exists():
        st.download_button("Download customers_for_powerbi.csv", Path(EXPORT_PATH).read_bytes(), "customers_for_powerbi.csv", "text/csv")
    st.caption("One flat table: customers joined with out-of-fold churn probabilities.")
