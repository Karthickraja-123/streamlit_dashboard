import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dynamic_pricing_preprocessed (2).csv")

df = load_data()


# ---------------------------------
# SECTION 1 — KPI DASHBOARD
# ---------------------------------
st.title("📊 Dynamic Pricing — AI Dashboard (All-in-One)")

st.subheader("📌 Key Performance Indicators (KPI)")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Revenue", f"{df['revenue'].sum():,.2f}")
col2.metric("Avg Price", f"{df['price'].mean():.2f}")
col3.metric("Units Sold", f"{df['units_sold'].sum():,.0f}")
col4.metric("Total Visits", f"{df['visits'].sum():,.0f}")
col5.metric("Avg Conversion Rate", f"{df['conversion_rate'].mean():.2f}")


# ---------------------------------
# SECTION 2 — VISUALIZATIONS
# ---------------------------------
st.header("📈 Data Visualizations")

chart_type = st.selectbox(
    "Choose chart type:",
    ["Bar Chart", "Line Chart", "Histogram", "Area Chart",
     "Scatter Plot", "Heatmap", "Bubble Chart", "Choropleth Map"]
)

# BAR CHART
if chart_type == "Bar Chart":
    fig = px.bar(df, x="category", y="price", color="category", title="Price by Category")
    st.plotly_chart(fig, use_container_width=True)

# LINE CHART
elif chart_type == "Line Chart":
    fig = px.line(df, x="timestamp", y="price", title="Price Over Time")
    st.plotly_chart(fig, use_container_width=True)

# HISTOGRAM
elif chart_type == "Histogram":
    fig = px.histogram(df, x="price", nbins=50, title="Price Distribution")
    st.plotly_chart(fig, use_container_width=True)

# AREA CHART
elif chart_type == "Area Chart":
    fig = px.area(df, x="timestamp", y="revenue", title="Revenue Over Time")
    st.plotly_chart(fig, use_container_width=True)

# SCATTER PLOT
elif chart_type == "Scatter Plot":
    fig = px.scatter(df, x="price", y="units_sold", color="category",
                     title="Price vs Units Sold (Demand Curve)")
    st.plotly_chart(fig, use_container_width=True)

# HEATMAP
elif chart_type == "Heatmap":
    st.write("### Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    st.pyplot(fig)

# BUBBLE CHART
elif chart_type == "Bubble Chart":
    fig = px.scatter(df, x="price", y="revenue", size="units_sold",
                     color="category", title="Bubble Chart (Revenue & Units Sold)")
    st.plotly_chart(fig, use_container_width=True)

# CHOROPLETH MAP (if location exists)
elif chart_type == "Choropleth Map":
    st.write("⚠ Your dataset does not contain geolocation. Add 'country' or 'state' column.")
    st.write("Showing demo chart below:")

    demo = pd.DataFrame({
        "country": ["India", "USA", "Canada"],
        "revenue": [25000, 18000, 9000]
    })

    fig = px.choropleth(demo, locations="country",
                        locationmode="country names",
                        color="revenue",
                        title="Demo Choropleth Map")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------
# SECTION 3 — ML MODEL METRICS
# ---------------------------------

st.header("🤖 Machine Learning — Model Performance")

uploaded = st.file_uploader("Upload your MODEL predictions CSV (y_test, y_pred)", type=["csv"])

if uploaded:
    model_df = pd.read_csv(uploaded)

    y_test = model_df["y_test"]
    y_pred = model_df["y_pred"]

    st.subheader("📌 Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.3f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")

    # CONFUSION MATRIX
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues")
    st.pyplot(fig)

    # ROC CURVE (if probabilities exist)
    if "p_low" in model_df.columns:
        st.subheader("📌 ROC Curve")

        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

        fig2, ax2 = plt.subplots(figsize=(6,5))
        for i, cls in enumerate(["Low", "Med", "High"]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], model_df[f"p_{cls.lower()}"])
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")

        ax2.plot([0,1], [0,1], "k--")
        ax2.legend()
        st.pyplot(fig2)

