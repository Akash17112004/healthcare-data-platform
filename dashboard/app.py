"""Interactive dashboard (Streamlit). Every number comes from the pipeline outputs - nothing is hard-coded.

    python run_pipeline.py            # build data/ first
    streamlit run dashboard/app.py
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

TEAL, AMBER, RED, GREY = "#1f6f8b", "#e0a030", "#c0392b", "#8a8f98"
AGE_ORDER = ["Child (0-12)", "Teen (13-17)", "Adult (18-39)", "Middle-aged (40-59)", "Senior (60+)"]
st.set_page_config(page_title="Smart Healthcare Analytics", page_icon="🏥", layout="wide")


@st.cache_data(show_spinner=False)
def load(folder: str) -> pd.DataFrame:
    path = Path(folder)
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# Live pipeline output if it exists, otherwise the committed snapshot (what the hosted dashboard uses)
DATA = C.DATA if (C.PREDICTIONS / "readmission_predictions").exists() else ROOT / "dashboard" / "data"
if not (DATA / "predictions" / "readmission_predictions").exists():
    st.error("No pipeline outputs found. Run `python run_pipeline.py` first, then reload this page.")
    st.stop()

ml = load(str(DATA / "gold" / "ml_dataset"))
alerts = load(str(DATA / "gold" / "vitals_alert_rates"))
pred = load(str(DATA / "predictions" / "readmission_predictions"))
comparison = load(str(DATA / "predictions" / "model_comparison.parquet"))
importance = load(str(DATA / "predictions" / "feature_importance.parquet"))
roc = load(str(DATA / "predictions" / "roc_curve.parquet"))
report = json.loads((C.REPORTS / "model_report.json").read_text())

# ── sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.title("Filters")
depts = sorted(ml["department"].unique())
sel_dept = st.sidebar.multiselect("Department", depts, default=depts)
types = sorted(ml["admission_type"].unique())
sel_type = st.sidebar.multiselect("Admission type", types, default=types)
lo, hi = ml["admit_ts"].min().date(), ml["admit_ts"].max().date()
sel_dates = st.sidebar.date_input("Admission date range", (lo, hi), min_value=lo, max_value=hi)
if len(sel_dates) != 2:
    st.sidebar.info("Pick an end date.")
    st.stop()
st.sidebar.caption("Simulated data - no real patients. Filters apply to every tab except model evaluation.")


def apply(df):
    d = df["admit_ts"].dt.date
    return df[df["department"].isin(sel_dept) & df["admission_type"].isin(sel_type) & (d >= sel_dates[0]) & (d <= sel_dates[1])]


f, p = apply(ml), apply(pred)
if f.empty:
    st.warning("No admissions match the current filters.")
    st.stop()

st.title("🏥 Smart Healthcare Analytics & Prediction Platform")
st.caption("Source systems → Bronze → Silver → PySpark processing → Gold → MLlib readmission model → this dashboard")

tab_ov, tab_dept, tab_vit, tab_model, tab_risk = st.tabs(
    ["Overview", "Departments", "Vitals alerts", "Readmission model", "Patient risk worklist"])

# ── Overview ────────────────────────────────────────────────────────────────
with tab_ov:
    c = st.columns(5)
    c[0].metric("Admissions", f"{len(f):,}")
    c[1].metric("Unique patients", f"{f['patient_id'].nunique():,}")
    c[2].metric("Avg length of stay", f"{f['length_of_stay_days'].mean():.1f} days")
    c[3].metric("Avg bill", f"₹{f['total_bill'].mean():,.0f}")
    c[4].metric("Mortality", f"{100 * (f['outcome'] == 'Deceased').mean():.1f}%")

    m = (f.groupby("admit_month").agg(admissions=("admission_id", "count"),
                                      readmission_pct=("is_readmission_30d", lambda s: 100 * s.mean())).reset_index())
    m["rolling_3m"] = m["admissions"].rolling(3, min_periods=1).mean()
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=m["admit_month"], y=m["admissions"], name="Admissions", marker_color=TEAL)
    fig.add_scatter(x=m["admit_month"], y=m["rolling_3m"], name="3-month average", line=dict(color=AMBER, width=3))
    fig.update_layout(title="Admissions per month", height=340, margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, width="stretch")
    m_r = m.iloc[1:]  # month 1 has no earlier history in the data, so it cannot contain readmissions
    fig = px.line(m_r, x="admit_month", y="readmission_pct", markers=True, title="Admissions that were 30-day readmissions (%)")
    fig.update_traces(line_color=RED)
    fig.update_layout(height=340, margin=dict(t=50, b=10), yaxis_title="%", xaxis_title=None, yaxis_rangemode="tozero")
    b.plotly_chart(fig, width="stretch")

    x, y = st.columns(2)
    mix = f.groupby(["age_group", "admission_type"]).size().reset_index(name="admissions")
    fig = px.bar(mix, x="age_group", y="admissions", color="admission_type", title="Admissions by age group and type",
                 color_discrete_sequence=[TEAL, AMBER, GREY], category_orders={"age_group": AGE_ORDER})
    fig.update_layout(height=340, margin=dict(t=50, b=10), xaxis_title=None)
    x.plotly_chart(fig, width="stretch")
    los = f.groupby("outcome").size().reset_index(name="admissions")
    fig = px.pie(los, names="outcome", values="admissions", hole=0.55, title="Outcomes",
                 color_discrete_sequence=[TEAL, AMBER, RED])
    fig.update_layout(height=340, margin=dict(t=50, b=10))
    y.plotly_chart(fig, width="stretch")

# ── Departments ─────────────────────────────────────────────────────────────
with tab_dept:
    d = (f.groupby("department").agg(admissions=("admission_id", "count"), avg_los=("length_of_stay_days", "mean"),
                                     avg_bill=("total_bill", "mean"), mortality_pct=("outcome", lambda s: 100 * (s == "Deceased").mean()),
                                     readmission_pct=("is_readmission_30d", lambda s: 100 * s.mean()),
                                     avg_alerts=("total_alerts", "mean")).reset_index().sort_values("admissions", ascending=False))
    a, b = st.columns(2)
    fig = px.bar(d.sort_values("avg_bill"), x="avg_bill", y="department", orientation="h", title="Average bill per admission (₹)",
                 color_discrete_sequence=[TEAL])
    fig.update_layout(height=380, margin=dict(t=50, b=10), yaxis_title=None, xaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    fig = px.bar(d.sort_values("avg_los"), x="avg_los", y="department", orientation="h", title="Average length of stay (days)",
                 color_discrete_sequence=[AMBER])
    fig.update_layout(height=380, margin=dict(t=50, b=10), yaxis_title=None, xaxis_title=None)
    b.plotly_chart(fig, width="stretch")
    heat = f.assign(month=f["admit_month"].astype(str)).pivot_table(index="department", columns="month", values="admission_id", aggfunc="count", fill_value=0)
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Teal", title="Admissions: department × month")
    fig.update_layout(height=360, margin=dict(t=50, b=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(d.round(2), hide_index=True, width="stretch")

# ── Vitals alerts ───────────────────────────────────────────────────────────
with tab_vit:
    st.caption("Share of bedside readings that triggered each clinical alert, by department "
               "(tachycardia HR>100, hypoxemia SpO₂<92, high BP ≥140/90, fever ≥38 °C, hyperglycaemia ≥180 mg/dL).")
    av = alerts[alerts["department"].isin(sel_dept)].set_index("department").drop(columns="readings")
    av.columns = [c.replace("_pct", "").replace("_", " ") for c in av.columns]
    fig = px.imshow(av, text_auto=".1f", aspect="auto", color_continuous_scale="OrRd", labels=dict(color="% of readings"))
    fig.update_layout(height=420, margin=dict(t=20, b=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

# ── Model ───────────────────────────────────────────────────────────────────
with tab_model:
    t = report["test_at_threshold"]
    st.subheader(f"30-day readmission classifier - {report['selected_model']} (Spark MLlib)")
    c = st.columns(5)
    c[0].metric("Test ROC-AUC", f"{report['test_roc_auc']:.3f}", help="0.5 = random guessing, 1.0 = perfect")
    c[1].metric("Test PR-AUC", f"{report['test_pr_auc']:.3f}", delta=f"{report['test_pr_auc'] - report['test_pr_auc_baseline']:+.3f} vs base rate")
    c[2].metric("Precision", f"{t['precision']:.2f}")
    c[3].metric("Recall", f"{t['recall']:.2f}")
    c[4].metric("Top-20% lift", f"{report['top20pct_lift']:.2f}×",
                help=f"The 20% highest-risk patients contain {100 * report['top20pct_capture_of_readmissions']:.0f}% of all readmissions")
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=roc["fpr"], y=roc["tpr"], name="Model", line=dict(color=TEAL, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=GREY, dash="dash"))
    fig.update_layout(title="ROC curve (held-out test patients)", height=360, margin=dict(t=50, b=10),
                      xaxis_title="False positive rate", yaxis_title="True positive rate", legend=dict(orientation="h", y=-0.25))
    a.plotly_chart(fig, width="stretch")
    top = importance.head(10).iloc[::-1]
    fig = px.bar(top, x="importance", y="feature", orientation="h", title="Most influential features", color_discrete_sequence=[TEAL])
    fig.update_layout(height=360, margin=dict(t=50, b=10), yaxis_title=None, xaxis_title=None)
    b.plotly_chart(fig, width="stretch")

    test = pred[pred["split"] == "test"]
    cal = (test.groupby("risk_band").agg(patients=("admission_id", "count"), actual_rate=("actual_readmitted_30d", lambda s: 100 * s.mean()))
           .reindex(["Low", "Medium", "High"]).reset_index())
    a, b = st.columns(2)
    fig = px.bar(cal, x="risk_band", y="actual_rate", text=cal["actual_rate"].round(1).astype(str) + "%", title="Actual readmission rate by predicted risk band (test)",
                 color="risk_band", color_discrete_map={"Low": TEAL, "Medium": AMBER, "High": RED})
    fig.update_layout(height=340, margin=dict(t=50, b=10), showlegend=False, yaxis_title="Actual readmitted (%)", xaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    cm = pd.DataFrame([[t["tn"], t["fp"]], [t["fn"], t["tp"]]], index=["Actual: no", "Actual: yes"], columns=["Predicted: no", "Predicted: yes"])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Teal", title=f"Confusion matrix (test set, threshold {t['threshold']})")
    fig.update_layout(height=340, margin=dict(t=50, b=10), coloraxis_showscale=False)
    b.plotly_chart(fig, width="stretch")
    st.markdown("**Model comparison** (selected on validation PR-AUC)")
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.info(report["label_note"])

# ── Risk worklist ───────────────────────────────────────────────────────────
with tab_risk:
    st.caption("Discharged in the last 30 days of the data snapshot: the follow-up window is still open, so the outcome is unknown "
               "and the model's score is a genuine forecast. Highest risk first.")
    cutoff = pd.Timestamp(C.REF_DATE - timedelta(days=30))
    open_window = p[(p["discharge_ts"] > cutoff) & (p["outcome"] != "Deceased")]
    bands = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High"])
    work = open_window[open_window["risk_band"].isin(bands)].sort_values("probability", ascending=False)
    st.metric("Patients to follow up", f"{len(work):,}", help=f"of {len(open_window):,} recently discharged in the current filter")
    show = work[["patient_id", "admission_id", "department", "admission_type", "age", "gender", "severity", "discharge_ts", "probability", "risk_band"]]
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"probability": st.column_config.ProgressColumn("Readmission risk", min_value=0, max_value=1, format="%.2f")})
    st.download_button("Download worklist (CSV)", show.to_csv(index=False).encode(), "followup_worklist.csv", "text/csv")
