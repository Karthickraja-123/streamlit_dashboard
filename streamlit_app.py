import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="Dynamic Pricing — Expert Dashboard", layout="wide", page_icon="📈")

# HEADER / TITLE
st.title("📈 Dynamic Pricing Optimization — Expert Dashboard")
st.markdown(
    "A professional, explanatory dashboard to explore pricing data, visualize patterns, "
    "and evaluate model performance. Use the sidebar to navigate sections and see "
    "contextual help and definitions throughout."
)

# SIDEBAR NAV
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", [
    "Overview & KPIs",
    "Dataset Explorer",
    "Visualizations",
    "Model Performance",
    "Data Dictionary",
    "Methodology & Notes",
    "README / Download"
])

# Load data (must be in same folder or update path)
@st.cache_data
def load_data(path="dynamic_pricing_preprocessed (2).csv"):
    df = pd.read_csv(path)
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Could not load dataset. Make sure 'dynamic_pricing_preprocessed (2).csv' is in the app folder.")
    st.stop()

# Helpful small utilities
def safe_mean(series):
    try:
        return round(series.mean(), 2)
    except:
        return "N/A"

# SECTION: Overview & KPIs
if section == "Overview & KPIs":
    st.header("Overview & Key Performance Indicators")
    st.markdown("""
    **What this page shows:** high-level KPIs derived from your dataset to quickly understand scale,
    central tendency and business impact. Hover over KPI titles for definitions.
    """)
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)

    total_revenue = df['revenue'].sum() if 'revenue' in df.columns else np.nan
    avg_price = safe_mean(df['price']) if 'price' in df.columns else "N/A"
    total_units = int(df['units_sold'].sum()) if 'units_sold' in df.columns else "N/A"
    total_visits = int(df['visits'].sum()) if 'visits' in df.columns else "N/A"
    avg_conv = safe_mean(df['conversion_rate']) if 'conversion_rate' in df.columns else "N/A"

    with col1:
        st.metric("Total Revenue", f"{total_revenue:,.2f}")
        st.caption("Total revenue calculated as sum(price × quantity).")
    with col2:
        st.metric("Average Price", avg_price)
        st.caption("Average selling price across dataset.")
    with col3:
        st.metric("Units Sold", total_units)
        st.caption("Total units sold — indicates volume.")
    with col4:
        st.metric("Total Visits", total_visits)
        st.caption("Sum of product page visits (traffic metric).")
    with col5:
        st.metric("Avg Conversion Rate", avg_conv)
        st.caption("Average conversion rate = purchases / visits (where provided).")

    st.markdown("### Business Signals and Quick Guidance")
    st.info("""
    • High revenue + low conversion could mean high price sensitivity.  
    • High visits + low units_sold suggests product listing or price issues.  
    • Use the Visualizations page for time-series and category deep-dives.
    """)

# SECTION: Dataset Explorer
if section == "Dataset Explorer":
    st.header("Dataset Explorer")
    st.markdown("Inspect the raw data and aggregated summaries. Use filtering to slice data for focused analysis.")
    with st.expander("Table preview & download"):
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download dataset CSV", csv, "dynamic_pricing_preprocessed (2).csv", "text/csv")
    with st.expander("Summary statistics (numeric)"):
        st.dataframe(df.select_dtypes(include=[np.number]).describe().T)
    with st.expander("Column presence & basic types"):
        col_info = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": [int(df[c].notnull().sum()) for c in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

# SECTION: Visualizations
if section == "Visualizations":
    st.header("Charts & Visualizations")
    st.markdown("Choose charts and filters. Descriptions appear under each chart to explain interpretation.")
    st.sidebar.subheader("Visualization Filters")
    cat_filter = None
    if 'category' in df.columns:
        cats = ["All"] + sorted(df['category'].dropna().unique().tolist())
        cat_filter = st.sidebar.selectbox("Filter by category", options=cats)
        if cat_filter != "All":
            vdf = df[df['category'] == cat_filter]
        else:
            vdf = df.copy()
    else:
        vdf = df.copy()

    chart = st.selectbox("Select chart type", [
        "Bar: Avg Price by Category",
        "Line: Price over Time",
        "Histogram: Price distribution",
        "Area: Revenue over Time",
        "Scatter: Price vs Units Sold",
        "Heatmap: Feature Correlation",
        "Bubble: Price vs Revenue (size=units_sold)"
    ])

    if chart == "Bar: Avg Price by Category":
        if 'category' in vdf.columns and 'price' in vdf.columns:
            agg = vdf.groupby('category')['price'].mean().reset_index().sort_values('price', ascending=False)
            fig = px.bar(agg, x='category', y='price', title="Average Price by Category", text='price')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Interpretation:** Bars show which categories have higher average prices. High-price categories could be premium lines or overpriced items.")
        else:
            st.warning("Need 'category' and 'price' columns for this chart.")

    if chart == "Line: Price over Time":
        if 'timestamp' in vdf.columns and 'price' in vdf.columns:
            tdf = vdf.copy()
            tdf['timestamp'] = pd.to_datetime(tdf['timestamp'])
            tdf = tdf.sort_values('timestamp').groupby(pd.Grouper(key='timestamp', freq='D'))['price'].mean().reset_index()
            fig = px.line(tdf, x='timestamp', y='price', title="Daily Average Price")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Interpretation:** Use to discover trends, seasonality, and sudden price shifts.")
        else:
            st.warning("Need 'timestamp' and 'price' columns for time series.")

    if chart == "Histogram: Price distribution":
        if 'price' in vdf.columns:
            fig = px.histogram(vdf, x='price', nbins=40, title="Price Distribution")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Interpretation:** Shows skew, mode(s), and outliers in price.")
        else:
            st.warning("Need 'price' column.")

    if chart == "Area: Revenue over Time":
        if 'timestamp' in vdf.columns and 'revenue' in vdf.columns:
            tdf = vdf.copy()
            tdf['timestamp'] = pd.to_datetime(tdf['timestamp'])
            tdf = tdf.sort_values('timestamp').groupby(pd.Grouper(key='timestamp', freq='D'))['revenue'].sum().reset_index()
            fig = px.area(tdf, x='timestamp', y='revenue', title="Daily Revenue")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Interpretation:** Helps spot revenue spikes and impacts of promotions.")
        else:
            st.warning("Need 'timestamp' and 'revenue' columns.")

    if chart == "Scatter: Price vs Units Sold":
        if 'price' in vdf.columns and 'units_sold' in vdf.columns:
            color = 'category' if 'category' in vdf.columns else None
            fig = px.scatter(vdf, x='price', y='units_sold', color=color, title="Price vs Units Sold")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Interpretation:** Downward trend indicates price elasticity; denser points show typical price range.")
        else:
            st.warning("Need 'price' and 'units_sold' columns.")

    if chart == "Heatmap: Feature Correlation":
        corr = vdf.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu", ax=ax)
        st.pyplot(fig)
        st.markdown("**Interpretation:** Correlations suggest relationships — e.g., price & revenue should be strongly correlated.")

    if chart == "Bubble: Price vs Revenue (size=units_sold)":
        if set(['price','revenue','units_sold']).issubset(vdf.columns):
            fig = px.scatter(vdf, x='price', y='revenue', size='units_sold', color='category' if 'category' in vdf.columns else None,
                             title="Bubble: Price vs Revenue")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Interpretation:** Bubbles show where high revenue & high volume overlap with price.")

# SECTION: Model Performance
if section == "Model Performance":
    st.header("Model Performance & Evaluation")
    st.markdown("Upload model prediction outputs (y_test, y_pred). Optionally include probability columns p_low/p_med/p_high for ROC curves.")
    uploaded = st.file_uploader("Upload CSV with columns: y_test, y_pred, (optional p_low,p_med,p_high)", type=['csv'])
    if uploaded:
        mdf = pd.read_csv(uploaded)
        if set(['y_test','y_pred']).issubset(mdf.columns):
            y_test = mdf['y_test']
            y_pred = mdf['y_pred']
            st.subheader("Classification Metrics")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c3.metric("Recall", f"{rec:.3f}")
            c4.metric("F1 Score", f"{f1:.3f}")
            st.markdown("**Classification Report**")
            st.text(classification_report(y_test, y_pred, zero_division=0))
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
            if set(['p_low','p_med','p_high']).issubset(mdf.columns):
                st.markdown("**ROC Curve (Multiclass)**")
                y_test_bin = label_binarize(y_test, classes=[0,1,2])
                probs = mdf[['p_low','p_med','p_high']].values
                fig2, ax2 = plt.subplots(figsize=(6,5))
                for i, cls in enumerate(['Low','Med','High']):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax2.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
                ax2.plot([0,1],[0,1],'k--')
                ax2.set_xlim(0,1)
                ax2.set_ylim(0,1)
                ax2.legend()
                st.pyplot(fig2)
        else:
            st.error("CSV must contain 'y_test' and 'y_pred' columns.")

# SECTION: Data Dictionary
if section == "Data Dictionary":
    st.header("Data Dictionary & Feature Descriptions")
    st.markdown("Detailed descriptions of columns to help non-technical stakeholders understand the dataset.")
    dict_rows = []
    # Provide descriptions for known columns; keep generic otherwise
    known = {
        "id": "Unique record identifier",
        "timestamp": "Date & time of the event/transaction",
        "product_id": "Product identifier",
        "category": "Product category",
        "base_price": "Base price before discount",
        "price": "Final selling price",
        "competitor_price": "Competitor's price for the same product",
        "promotion_flag": "1 if promotion active, else 0",
        "units_sold": "Units sold in that record interval",
        "visits": "Number of visits/pageviews",
        "conversion_rate": "units_sold / visits",
        "stock_level": "Available stock",
        "revenue": "price × units_sold",
        "price_diff": "price - competitor_price",
        "discount_percent": "Percentage discount applied",
        "revenue_per_visit": "revenue / visits",
        "hour": "Hour of day (0-23)",
        "day": "Day of month",
        "weekday": "Day of week (0=Mon)",
        "month": "Month (1-12)",
        "price_category": "Binned label (low/med/high)"
    }
    for c in df.columns:
        dict_rows.append({"column": c, "description": known.get(c, "—")})
    st.dataframe(pd.DataFrame(dict_rows), use_container_width=True)

# SECTION: Methodology & Notes
if section == "Methodology & Notes":
    st.header("Methodology, Assumptions & Next Steps")
    st.markdown("""
    **How targets were created (example)**  
    - `price_category` was created via quantile binning into Low / Med / High price groups.  
    **Assumptions**  
    - Timestamps are in local timezone.  
    - Missing values were handled during preprocessing (impute or drop).  
    **Recommended next steps (expert):**  
    1. Calibrate price elasticity per product using demand curves.  
    2. A/B test model-driven price recommendations on a subset of SKUs.  
    3. Add customer segmentation and CLV module to prioritize high-value segments.
    """)
    with st.expander("Modeling tips for data scientists"):
        st.markdown("""
        • Use stratified sampling for class imbalance.  
        • Monitor downstream KPIs: Revenue uplift, conversion delta, churn.  
        • Maintain experiment tracking (weights, datasets, hyperparams).
        """)

# SECTION: README / Download
if section == "README / Download":
    st.header("Repository README & Downloads")
    st.markdown("Download README.md or the refined Streamlit script for GitHub upload.")
    readme_text = """
# Dynamic Pricing — Expert Dashboard

This repository contains a Streamlit app for exploring dynamic pricing data, generating business KPIs,
visualizations, and model evaluation metrics.

## Contents
- streamlit_app.py (this dashboard)
- dynamic_pricing_preprocessed (2).csv (dataset)
- requirements.txt

## How to run
1. Install dependencies: pip install -r requirements.txt  
2. Run locally: streamlit run streamlit_app.py  
3. For Streamlit Cloud: push repo to GitHub and deploy.

## Features
- KPI cards, interactive charts, dataset explorer
- Upload model predictions for evaluation (ROC / Confusion matrix)
- Data dictionary and methodology notes
"""
    st.download_button("Download README.md", readme_text, "README.md", "text/markdown")
    st.download_button("Download app script", open(__file__, "r", encoding="utf-8").read(), "streamlit_app.py", "text/x-python")

