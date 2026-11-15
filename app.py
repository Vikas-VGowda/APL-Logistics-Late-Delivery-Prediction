import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------
# Page configuration and paths
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="APL Logistics | Late Delivery Risk",
    page_icon="🚚",
    layout="wide",
)

# Public Hugging Face repository containing the trained model artifacts.
HF_REPO_ID = "Vvg0007/apl-logistics-model"
MODEL_FILENAME = "best_late_delivery_model.joblib"
FEATURES_FILENAME = "feature_columns.joblib"


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
      .main-header {
        padding: 0.35rem 0 0.15rem 0;
      }
      div[data-testid="stMetric"] {
        background: #f4f7fb;
        border: 1px solid #e5eaf2;
        padding: 14px 16px;
        border-radius: 12px;
      }
      .section-note {
        color: #64748b;
        font-size: 0.9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Download and load the trained pipeline and expected feature list.
# Files are cached by Streamlit after the first successful download.
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading trained model from Hugging Face...")
def load_artifacts():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
    )
    features_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=FEATURES_FILENAME,
        repo_type="model",
    )
    fitted_pipeline = joblib.load(model_path)
    feature_list = joblib.load(features_path)
    return fitted_pipeline, list(feature_list)


@st.cache_data(show_spinner=False)
def read_csv_file(file_bytes: bytes) -> pd.DataFrame:
    # The supplied APL dataset was read with cp1252 in the Colab notebook.
    try:
        from io import BytesIO
        return pd.read_csv(BytesIO(file_bytes), encoding="cp1252")
    except UnicodeDecodeError:
        from io import BytesIO
        return pd.read_csv(BytesIO(file_bytes), encoding="latin1")


def risk_label(probability: float, high_threshold: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability < high_threshold:
        return "Medium Risk"
    return "High Risk"


def safe_display_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


# ---------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🚚 APL Logistics")
st.subheader("Machine Learning-Based Late Delivery Risk Prediction")
st.caption(
    "Predictive analytics to help logistics teams identify orders that may "
    "require early attention."
)
st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Dashboard controls")
    st.markdown("Upload the order dataset to generate predictions.")
    uploaded_file = st.file_uploader(
        "APL Logistics CSV",
        type=["csv"],
        help=(
            "Upload the CSV used for this project. The app uses only the "
            "pre-shipment features expected by the trained model."
        ),
    )
    st.divider()
    high_threshold = st.slider(
        "High-risk probability threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help=(
            "Orders at or above this predicted probability are classified "
            "as High Risk. Probabilities below 0.30 are Low Risk."
        ),
    )

try:
    model, feature_columns = load_artifacts()
except Exception as exc:
    st.error(
        "The model could not be downloaded or loaded from Hugging Face. "
        "Check that the repository is public, both artifact filenames "
        "match exactly, and the installed library versions are compatible "
        "with the Colab training environment."
    )
    st.exception(exc)
    st.stop()

if uploaded_file is None:
    st.info("Upload `APL_Logistics.csv` using the sidebar to open the dashboard.")
    st.markdown(
        """
        **Dashboard modules**
        1. Delay Risk Overview
        2. Order-Level Risk Prediction
        3. Region & Shipping-Mode Risk Analysis
        4. Operations Action Panel

        The original dataset is intentionally not bundled into this app.
        Avoid committing customer names, street addresses, ZIP codes, or
        other personal information to a public GitHub repository.
        """
    )
    st.stop()

try:
    raw_bytes = uploaded_file.getvalue()
    df = read_csv_file(raw_bytes)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    st.error(
        "This CSV does not contain all columns required by the trained model. "
        "Missing columns: " + ", ".join(missing_features)
    )
    st.stop()

if df.empty:
    st.warning("The uploaded CSV contains no rows.")
    st.stop()

# Match the exact feature order used when the model was trained.
X = df.loc[:, feature_columns].copy()

try:
    with st.spinner("Scoring orders with the trained model..."):
        probabilities = model.predict_proba(X)[:, 1]
except Exception as exc:
    st.error(
        "Prediction failed. Check that the uploaded dataset has the same "
        "column formats as the training dataset and that package versions "
        "match the Colab environment."
    )
    st.exception(exc)
    st.stop()

results = df.copy()
results["Predicted Probability"] = probabilities
results["Risk Category"] = [
    risk_label(float(p), high_threshold) for p in probabilities
]

# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Filters")

filtered = results.copy()

filter_columns = [
    ("Shipping Mode", "Shipping mode"),
    ("Order Region", "Order region"),
    ("Customer Segment", "Customer segment"),
]

for column, label in filter_columns:
    if column in filtered.columns:
        options = sorted(
            filtered[column].dropna().astype(str).unique().tolist()
        )
        selected = st.sidebar.multiselect(
            label,
            options,
            default=options,
            key=f"filter_{column}",
        )
        filtered = filtered[filtered[column].astype(str).isin(selected)]

# ---------------------------------------------------------------------
# Module 1: Delay Risk Overview
# ---------------------------------------------------------------------
st.divider()
st.header("1. Delay Risk Overview")

total_orders = len(filtered)
high_count = int((filtered["Risk Category"] == "High Risk").sum())
medium_count = int((filtered["Risk Category"] == "Medium Risk").sum())
low_count = int((filtered["Risk Category"] == "Low Risk").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Orders in view", f"{total_orders:,}")
m2.metric("High-risk orders", f"{high_count:,}")
m3.metric("Medium-risk orders", f"{medium_count:,}")
m4.metric("Low-risk orders", f"{low_count:,}")

if not filtered.empty:
    risk_order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_counts = (
        filtered["Risk Category"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("Risk Category")
        .reset_index(name="Orders")
    )

    left, right = st.columns(2)
    with left:
        fig = px.pie(
            risk_counts,
            names="Risk Category",
            values="Orders",
            hole=0.48,
            title="Predicted Risk Distribution",
            color="Risk Category",
            color_discrete_map={
                "Low Risk": "#16a34a",
                "Medium Risk": "#f59e0b",
                "High Risk": "#dc2626",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(
            filtered,
            x="Predicted Probability",
            nbins=30,
            title="Predicted Late-Delivery Probability",
            labels={"Predicted Probability": "Predicted probability"},
        )
        fig.add_vline(
            x=high_threshold,
            line_dash="dash",
            annotation_text="High-risk threshold",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No orders match the selected filters.")

# ---------------------------------------------------------------------
# Module 2: Order-Level Risk Prediction
# ---------------------------------------------------------------------
st.divider()
st.header("2. Order-Level Risk Prediction")

if not filtered.empty:
    order_id_col = "Order Id" if "Order Id" in filtered.columns else None
    row_indices = filtered.index.tolist()

    def order_option(index):
        if order_id_col:
            return f"Order ID {filtered.loc[index, order_id_col]} (row {index})"
        return f"Dataset row {index}"

    selected_index = st.selectbox(
        "Choose an order",
        options=row_indices,
        format_func=order_option,
    )

    selected_row = filtered.loc[selected_index]
    probability = float(selected_row["Predicted Probability"])
    category = selected_row["Risk Category"]

    p1, p2 = st.columns(2)
    p1.metric("Predicted late-delivery probability", f"{probability:.1%}")
    p2.metric("Predicted risk category", category)
    st.progress(min(max(probability, 0.0), 1.0))

    order_detail_columns = safe_display_columns(
        selected_row.to_frame().T,
        [
            "Order Id",
            "Shipping Mode",
            "Order Region",
            "Order Country",
            "Customer Segment",
            "Days for shipment (scheduled)",
            "Category Name",
            "Market",
        ],
    )
    if order_detail_columns:
        st.dataframe(
            selected_row[order_detail_columns].to_frame("Order details"),
            use_container_width=True,
        )

    st.caption(
        "This is a model-estimated probability, not a confirmed delivery "
        "outcome or a guarantee that an order will be late."
    )
else:
    st.info("Select filters that return at least one order.")

# ---------------------------------------------------------------------
# Module 3: Region & Mode Risk Analysis
# ---------------------------------------------------------------------
st.divider()
st.header("3. Region & Mode Risk Analysis")

if not filtered.empty:
    chart_columns = st.columns(2)

    if "Order Region" in filtered.columns:
        region_summary = (
            filtered.groupby("Order Region", dropna=False)
            ["Predicted Probability"]
            .mean()
            .mul(100)
            .reset_index(name="Average predicted risk (%)")
            .sort_values("Average predicted risk (%)", ascending=False)
        )
        with chart_columns[0]:
            fig = px.bar(
                region_summary,
                x="Order Region",
                y="Average predicted risk (%)",
                title="Average Predicted Risk by Region",
            )
            st.plotly_chart(fig, use_container_width=True)

    if "Shipping Mode" in filtered.columns:
        mode_summary = (
            filtered.groupby("Shipping Mode", dropna=False)
            ["Predicted Probability"]
            .mean()
            .mul(100)
            .reset_index(name="Average predicted risk (%)")
            .sort_values("Average predicted risk (%)", ascending=False)
        )
        with chart_columns[1]:
            fig = px.bar(
                mode_summary,
                x="Shipping Mode",
                y="Average predicted risk (%)",
                title="Average Predicted Risk by Shipping Mode",
            )
            st.plotly_chart(fig, use_container_width=True)

    if "Customer Segment" in filtered.columns:
        segment_summary = (
            filtered.groupby("Customer Segment", dropna=False)
            ["Predicted Probability"]
            .mean()
            .mul(100)
            .reset_index(name="Average predicted risk (%)")
            .sort_values("Average predicted risk (%)", ascending=False)
        )
        fig = px.bar(
            segment_summary,
            x="Customer Segment",
            y="Average predicted risk (%)",
            title="Average Predicted Risk by Customer Segment",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Module 4: Operations Action Panel
# ---------------------------------------------------------------------
st.divider()
st.header("4. Operations Action Panel")

high_risk_orders = (
    filtered[filtered["Risk Category"] == "High Risk"]
    .sort_values("Predicted Probability", ascending=False)
    .copy()
)

st.write(f"**Orders requiring review:** {len(high_risk_orders):,}")

action_columns = safe_display_columns(
    high_risk_orders,
    [
        "Order Id",
        "Shipping Mode",
        "Order Region",
        "Customer Segment",
        "Days for shipment (scheduled)",
        "Predicted Probability",
        "Risk Category",
    ],
)

if not high_risk_orders.empty:
    st.dataframe(
        high_risk_orders[action_columns],
        use_container_width=True,
        hide_index=True,
    )

    # Export only operationally useful fields, not all customer PII.
    export = high_risk_orders[action_columns].copy()
    csv_data = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download High-Risk Order Queue",
        data=csv_data,
        file_name="apl_high_risk_order_queue.csv",
        mime="text/csv",
    )
else:
    st.success("No orders are currently classified as High Risk.")

st.markdown(
    """
    **Suggested operational responses**
    - **High Risk:** Prioritize a shipment review and investigate possible
      bottlenecks or rerouting options.
    - **Medium Risk:** Monitor shipment progress and consider preventive
      communication or planning.
    - **Low Risk:** Continue normal operational monitoring.
    """
)

# ---------------------------------------------------------------------
# Model interpretation and footer
# ---------------------------------------------------------------------
st.divider()
st.header("Model Insights")

try:
    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocessing")

    if (
        classifier is not None
        and preprocessor is not None
        and hasattr(classifier, "feature_importances_")
    ):
        names = preprocessor.get_feature_names_out()
        values = classifier.feature_importances_

        importance = (
            pd.DataFrame({"Feature": names, "Importance": values})
            .sort_values("Importance", ascending=False)
            .head(15)
            .sort_values("Importance")
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Global Feature Importances",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Feature importance is not available for the loaded model "
            "in the expected pipeline format."
        )
except Exception as exc:
    st.warning(f"Could not display feature importance: {exc}")

st.caption(
    "APL Logistics | Late Delivery Risk Prediction | "
    "Predictions are decision-support estimates and should be reviewed "
    "by operations personnel."
)
