import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = r'''# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
'''


st.set_page_config(
    page_title="Mini Project B — Time-Series Forecasting Starter",
    page_icon="📈",
    layout="wide",
)


def get_openrouter_api_key():
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        key = ""
    if key:
        return key
    key = os.getenv("OPENROUTER_API_KEY", "")
    if key:
        return key
    return st.session_state.get("openrouter_api_key", "")


def audit_dataframe(dataframe):
    audit = pd.DataFrame({
        "column": dataframe.columns,
        "dtype": [str(dataframe[col].dtype) for col in dataframe.columns],
        "missing_percent": [float(dataframe[col].isna().mean() * 100) for col in dataframe.columns],
        "unique_count": [int(dataframe[col].nunique(dropna=True)) for col in dataframe.columns],
    })
    return audit


def clean_time_series(dataframe, timestamp_column, target_column):
    cleaned = dataframe.copy()
    cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], errors="coerce")
    cleaned[target_column] = pd.to_numeric(cleaned[target_column], errors="coerce")
    before_rows = len(cleaned)
    cleaned = cleaned.dropna(subset=[timestamp_column, target_column]).sort_values(timestamp_column)
    cleaned = cleaned.drop_duplicates(subset=[timestamp_column], keep="last")
    cleaned = cleaned.reset_index(drop=True)
    return cleaned, before_rows - len(cleaned)


def resample_series(dataframe, timestamp_column, target_column, rule):
    if rule == "None":
        return dataframe[[timestamp_column, target_column]].copy()
    temp = dataframe[[timestamp_column, target_column]].copy()
    temp = temp.set_index(timestamp_column).sort_index()
    temp = temp[target_column].resample(rule).mean().dropna().reset_index()
    return temp


def build_baseline_features(dataframe, timestamp_column, target_column, horizon):
    temp = dataframe[[timestamp_column, target_column]].copy().sort_values(timestamp_column)
    temp["lag_1"] = temp[target_column].shift(1)
    temp["lag_24"] = temp[target_column].shift(24)
    temp["rolling_mean_24"] = temp[target_column].shift(1).rolling(window=24, min_periods=24).mean()
    temp["hour"] = temp[timestamp_column].dt.hour
    temp["weekend"] = temp[timestamp_column].dt.dayofweek.isin([5, 6]).astype(int)
    temp["month"] = temp[timestamp_column].dt.month
    temp["y_target"] = temp[target_column].shift(-int(horizon))

    feature_columns = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
    feature_table = temp.dropna(subset=feature_columns + ["y_target"]).reset_index(drop=True)
    X = feature_table[feature_columns]
    y = feature_table["y_target"]
    return feature_table, X, y, feature_columns


def dataframe_to_records(value):
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict(orient="records")
    return []


def parse_ai_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def call_openrouter(prompt, api_key):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Mini Project B Forecasting Grader",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption("Starter app for dataset audit, baseline feature preparation, exports, and AI grading evidence.")

with st.sidebar:
    st.header("Student information")
    student_name = st.text_input("Student name", value="Al Azhar Hilal Al Azri")
    student_id = st.text_input("Student ID", value="PG12S2540471")
    deployed_url = st.text_input("Deployed Streamlit URL", value="")
    project_title = st.text_input("Project title", value="PJME Hourly Energy Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Prepare a time-series forecasting dashboard for hourly PJME electricity demand.",
        height=100,
    )
    st.header("Dataset path")
    dataset_path = st.text_input("Local dataset path", value="data/dataset_sample.csv")
    st.header("AI grader key")
    st.session_state["openrouter_api_key"] = st.text_input(
        "OpenRouter API key",
        value="",
        type="password",
        help="Used only if Streamlit Secrets and environment variable are not available.",
    )

st.subheader("1. Load dataset")
try:
    df = pd.read_csv(dataset_path)
except Exception as exc:
    st.error(f"Could not load dataset from {dataset_path}: {exc}")
    st.stop()

st.write(f"Rows: {len(df):,} | Columns: {df.shape[1]:,}")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. Dataset audit")
audit = audit_dataframe(df)
left, right = st.columns(2)
with left:
    st.write("Columns, dtypes, missing values, and unique counts")
    st.dataframe(audit, use_container_width=True)
with right:
    st.write("Top missing-value columns")
    st.dataframe(
        audit.sort_values("missing_percent", ascending=False).head(10),
        use_container_width=True,
    )

st.subheader("3. Timestamp and target selection")
columns = list(df.columns)
default_timestamp_index = columns.index("Datetime") if "Datetime" in columns else 0
numeric_like_columns = [
    col for col in columns
    if pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.75
]
default_target = "PJME_MW" if "PJME_MW" in columns else (numeric_like_columns[0] if numeric_like_columns else columns[0])
default_target_index = columns.index(default_target)

timestamp_column = st.selectbox("Timestamp column", columns, index=default_timestamp_index)
target_column = st.selectbox("Numeric target column", columns, index=default_target_index)

cleaned_df, dropped_rows = clean_time_series(df, timestamp_column, target_column)
if cleaned_df.empty:
    st.error("No valid rows remain after timestamp parsing and target conversion.")
    st.stop()

st.success(f"Cleaned rows: {len(cleaned_df):,} | Dropped invalid rows: {dropped_rows:,}")
coverage_start = cleaned_df[timestamp_column].min()
coverage_end = cleaned_df[timestamp_column].max()
st.write(f"Time coverage: {coverage_start} to {coverage_end}")

fig, ax = plt.subplots(figsize=(10, 3))
plot_df = cleaned_df[[timestamp_column, target_column]].tail(1000)
ax.plot(plot_df[timestamp_column], plot_df[target_column])
ax.set_title(f"Recent {target_column} values")
ax.set_xlabel(timestamp_column)
ax.set_ylabel(target_column)
st.pyplot(fig)

st.subheader("4. Optional resampling and forecast horizon")
resample_rule = st.selectbox(
    "Resampling",
    ["None", "H", "D", "W", "M"],
    help="None keeps original rows. H=hourly, D=daily, W=weekly, M=monthly mean.",
)
horizon = st.number_input("Forecast horizon in rows after resampling", min_value=1, max_value=168, value=24, step=1)

prepared_df = resample_series(cleaned_df, timestamp_column, target_column, resample_rule)
if len(prepared_df) < 50:
    st.warning("The prepared dataset is very small. Consider using less aggressive resampling.")

st.write(f"Prepared rows after resampling: {len(prepared_df):,}")

st.subheader("5. Baseline feature table")
feature_table, X, y, feature_columns = build_baseline_features(prepared_df, timestamp_column, target_column, horizon)
st.write(f"Feature rows: {len(feature_table):,} | Feature columns: {len(feature_columns)}")
st.dataframe(feature_table.head(20), use_container_width=True)

with st.expander("Baseline feature engineering details"):
    st.write("The starter app prepares these baseline features only:")
    st.write(feature_columns)
    st.write("Students should add their own models, metrics, plots, and insights below.")

st.subheader("6. STUDENT ADDITIONS — MODELING")
st.info("This section adds a time-based train/test split, two forecasting models, and a metrics table.")

# STUDENT ADDITIONS — MODELING
# Time-based train/test split + two forecasting models.
# This creates results_df so the export and AI grader can detect your metrics table.

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results_df = None
y_test = None
naive_pred = None
rf_pred = None
comparison_df = None
importance_df = None

if len(feature_table) > 100:
    model_df = feature_table.dropna().copy()

    X_model = model_df[feature_columns]
    y_model = model_df["y_target"]

    split_index = int(len(model_df) * 0.8)

    X_train = X_model.iloc[:split_index]
    X_test = X_model.iloc[split_index:]
    y_train = y_model.iloc[:split_index]
    y_test = y_model.iloc[split_index:]

    st.write("Training rows:", len(X_train))
    st.write("Testing rows:", len(X_test))

    # Model 1: Naive Lag-1 Baseline
    naive_pred = X_test["lag_1"]

    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    naive_r2 = r2_score(y_test, naive_pred)

    # Model 2: Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=12,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    # Metrics table
    results_df = pd.DataFrame([
        {
            "model": "Naive Lag-1 Baseline",
            "MAE": naive_mae,
            "RMSE": naive_rmse,
            "R2": naive_r2,
        },
        {
            "model": "Random Forest Regressor",
            "MAE": rf_mae,
            "RMSE": rf_rmse,
            "R2": rf_r2,
        },
    ])

    st.subheader("Model Metrics Table")
    st.dataframe(results_df, use_container_width=True)

    # Actual vs predicted comparison
    comparison_df = pd.DataFrame({
        "Actual": y_test.values,
        "Naive Prediction": naive_pred.values,
        "Random Forest Prediction": rf_pred,
    })

    st.subheader("Actual vs Predicted Forecast")
    st.line_chart(comparison_df.head(300))

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    st.subheader("Random Forest Feature Importance")
    st.dataframe(importance_df, use_container_width=True)
    st.bar_chart(importance_df.set_index("feature"))

    st.success("Modeling complete. results_df is ready for submission export.")

else:
    st.warning("Not enough rows for modeling after feature engineering.")


st.subheader("7. STUDENT ADDITIONS — DASHBOARD")
st.info("This section adds KPIs, forecast visuals, error analysis, and seasonal demand patterns.")

# STUDENT ADDITIONS — DASHBOARD
# Extra dashboard visuals and written evidence for the project.

if isinstance(results_df, pd.DataFrame) and len(results_df) > 0:
    st.subheader("Dashboard KPIs")

    best_model_row = results_df.sort_values("RMSE").iloc[0]
    best_model_name = best_model_row["model"]
    best_rmse = best_model_row["RMSE"]
    best_mae = best_model_row["MAE"]
    best_r2 = best_model_row["R2"]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Best Model", best_model_name)
    kpi2.metric("Best RMSE", f"{best_rmse:,.2f}")
    kpi3.metric("Best MAE", f"{best_mae:,.2f}")
    kpi4.metric("Best R²", f"{best_r2:.3f}")

    st.subheader("Model Ranking")
    ranked_results = results_df.sort_values("RMSE").reset_index(drop=True)
    st.dataframe(ranked_results, use_container_width=True)
    st.bar_chart(ranked_results.set_index("model")[["RMSE", "MAE"]])

    if comparison_df is not None:
        st.subheader("Forecast Error Analysis")
        error_df = comparison_df.copy()
        error_df["Random Forest Error"] = error_df["Actual"] - error_df["Random Forest Prediction"]
        error_df["Absolute Error"] = error_df["Random Forest Error"].abs()
        st.write("Largest Random Forest forecast errors in the test period.")
        st.dataframe(error_df.sort_values("Absolute Error", ascending=False).head(20), use_container_width=True)
        st.line_chart(error_df[["Random Forest Error"]].head(300))

else:
    st.warning("Run the modeling section first so dashboard KPIs can use the metrics table.")


st.subheader("Demand Pattern Dashboard")

pattern_df = cleaned_df[[timestamp_column, target_column]].copy()
pattern_df[timestamp_column] = pd.to_datetime(pattern_df[timestamp_column], errors="coerce")
pattern_df[target_column] = pd.to_numeric(pattern_df[target_column], errors="coerce")
pattern_df = pattern_df.dropna(subset=[timestamp_column, target_column])

pattern_df["hour"] = pattern_df[timestamp_column].dt.hour
pattern_df["day_of_week"] = pattern_df[timestamp_column].dt.day_name()
pattern_df["month"] = pattern_df[timestamp_column].dt.month
pattern_df["year"] = pattern_df[timestamp_column].dt.year

st.write("These charts show how electricity demand changes by time patterns.")

hourly_pattern = pattern_df.groupby("hour")[target_column].mean().reset_index()
st.subheader("Average Demand by Hour")
st.line_chart(hourly_pattern.set_index("hour"))

monthly_pattern = pattern_df.groupby("month")[target_column].mean().reset_index()
st.subheader("Average Demand by Month")
st.bar_chart(monthly_pattern.set_index("month"))

yearly_pattern = pattern_df.groupby("year")[target_column].mean().reset_index()
st.subheader("Average Demand by Year")
st.line_chart(yearly_pattern.set_index("year"))


st.subheader("Missing Values and Data Quality Evidence")

missing_summary = audit[["column", "missing_percent", "unique_count"]].copy()
st.dataframe(missing_summary, use_container_width=True)

missing_chart = missing_summary.set_index("column")[["missing_percent"]]
st.bar_chart(missing_chart)


st.subheader("Recent Demand Trend")

recent_trend = cleaned_df[[timestamp_column, target_column]].tail(24 * 14).copy()
recent_trend = recent_trend.set_index(timestamp_column)

st.write("Recent two-week demand trend based on the cleaned time-series data.")
st.line_chart(recent_trend)


dashboard_summary = """
Dashboard insights:
- The KPI cards compare models using RMSE, MAE, and R².
- The hourly chart shows daily electricity demand cycles.
- The monthly chart shows seasonal demand changes.
- The recent trend chart helps inspect short-term changes in demand.
- The data quality table documents missing values and unique counts.
"""

st.info(dashboard_summary)


student_insights = st.text_area(
    "Student insights and interpretation",
    value="",
    height=120,
    help="Write your final project insights after adding models and dashboard visuals.",
)

st.subheader("8. Export submission files")
has_metrics_table = isinstance(results_df, pd.DataFrame)
submission = {
    "student_name": student_name,
    "student_id": student_id,
    "deployed_url": deployed_url,
    "project_title": project_title,
    "project_goal": project_goal,
    "dataset_path": dataset_path,
    "timestamp_column": timestamp_column,
    "target_column": target_column,
    "raw_rows": int(len(df)),
    "cleaned_rows": int(len(cleaned_df)),
    "prepared_rows": int(len(prepared_df)),
    "feature_rows": int(len(feature_table)),
    "time_coverage_start": str(coverage_start),
    "time_coverage_end": str(coverage_end),
    "resampling": resample_rule,
    "forecast_horizon": int(horizon),
    "baseline_features": feature_columns,
    "has_feature_table": len(feature_table) > 0,
    "has_metrics_table": has_metrics_table,
    "results_table": dataframe_to_records(results_df),
    "student_insights": student_insights,
    "evidence_flags": {
        "timestamp_parsed": True,
        "target_numeric": True,
        "missing_values_audited": True,
        "resampling_option_available": True,
        "baseline_features_created": len(feature_table) > 0,
        "student_added_models": has_metrics_table,
        "student_added_dashboard": bool(student_insights.strip()),
    },
    "generated_at": datetime.utcnow().isoformat() + "Z",
}

submission_json = json.dumps(submission, indent=2)
project_card = f"""# {project_title}

Student: {student_name}  
Student ID: {student_id}

## Goal
{project_goal}

## Dataset
- Path: {dataset_path}
- Timestamp column: {timestamp_column}
- Target column: {target_column}
- Raw rows: {len(df):,}
- Cleaned rows: {len(cleaned_df):,}
- Time coverage: {coverage_start} to {coverage_end}

## Preparation
- Resampling: {resample_rule}
- Forecast horizon: {horizon}
- Baseline features: {", ".join(feature_columns)}

## Student insights
{student_insights if student_insights.strip() else "Add insights after completing modeling and dashboard work."}

## Submission links
- Streamlit URL: {deployed_url if deployed_url else "Add deployed URL"}
"""

col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download submission.json",
        data=submission_json,
        file_name="submission.json",
        mime="application/json",
    )
with col_b:
    st.download_button(
        "Download project_card.md",
        data=project_card,
        file_name="project_card.md",
        mime="text/markdown",
    )

st.subheader("9. AI grader out of 80")
st.write(f"Model: `{OPENROUTER_MODEL}`")
grader_prompt = AI_GRADER_PROMPT_TEMPLATE.replace("<insert submission.json contents here>", submission_json)

with st.expander("Preview grading evidence JSON"):
    st.code(submission_json, language="json")

if st.button("Run AI grader"):
    api_key = get_openrouter_api_key()
    if not api_key:
        st.error("OpenRouter API key is required. Add it to Streamlit Secrets, environment variable, or the password field.")
    else:
        try:
            raw_output = call_openrouter(grader_prompt, api_key)
            parsed = parse_ai_json(raw_output)
            if parsed is not None:
                st.success("AI grader returned valid JSON.")
                st.json(parsed)
            else:
                st.warning("Could not parse JSON. Raw model output is shown below.")
                st.code(raw_output)
        except Exception as exc:
            st.error(f"AI grader request failed: {exc}")
