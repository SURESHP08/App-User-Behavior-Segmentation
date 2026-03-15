# Advanced User Segmentation Dashboard


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Page Setup


st.set_page_config(
    page_title="User Behavior Analytics Dashboard",
    layout="wide"
)

st.title("📊 App User Behavior Analytics Dashboard")

st.markdown("""
This dashboard analyzes **user engagement behavior** using **K-Means clustering**.
The dataset contains **50,000 users segmented into 4 behavioral groups**.
""")


# Load Dataset


df = pd.read_csv("user_segmentation_results.csv")

features = [
    "sessions_per_week",
    "avg_session_duration_min",
    "daily_active_minutes",
    "engagement_score"
]


# Sidebar Filters


st.sidebar.header("🎛 Dashboard Filters")

cluster_filter = st.sidebar.multiselect(
    "Select Clusters",
    options=sorted(df["cluster"].unique()),
    default=sorted(df["cluster"].unique())
)

filtered_df = df[df["cluster"].isin(cluster_filter)]


# KPI Metrics


st.subheader("📈 Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Users", len(filtered_df))
col2.metric("Avg Sessions / Week", round(filtered_df["sessions_per_week"].mean(),2))
col3.metric("Avg Session Duration", round(filtered_df["avg_session_duration_min"].mean(),2))
col4.metric("Avg Engagement Score", round(filtered_df["engagement_score"].mean(),2))


# Cluster Interpretation Cards


st.subheader("🧠 Cluster Interpretation")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Cluster 0 – High Engagement Users**

    • Frequent sessions  
    • Long session durations  
    • High engagement score  

    Business Action: Loyalty programs & premium offers
    """)

    st.warning("""
    **Cluster 2 – Low Engagement / At-Risk**

    • Low activity  
    • Short sessions  

    Business Action: Retention campaigns
    """)

with col2:
    st.success("""
    **Cluster 1 – Moderate Users**

    • Balanced usage behavior  

    Business Action: Personalized recommendations
    """)

    st.success("""
    **Cluster 3 – Occasional Users**
    
    • Irregular interaction  

    Business Action: Engagement reminders
    """)


# Chart Row 1


st.subheader("📊 User Distribution Analysis")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots()

    cluster_counts = filtered_df["cluster"].value_counts()

    ax.bar(cluster_counts.index, cluster_counts.values)

    ax.set_title("Cluster Distribution")

    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots()

    sns.boxplot(
        data=filtered_df,
        x="cluster",
        y="engagement_score",
        palette="Set2",
        ax=ax
    )

    ax.set_title("Engagement Score by Cluster")

    st.pyplot(fig)


# Chart Row 2


st.subheader("📊 User Behavior Comparison")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots()

    sns.boxplot(
        data=filtered_df,
        x="cluster",
        y="sessions_per_week",
        palette="Set3",
        ax=ax
    )

    ax.set_title("Sessions Per Week by Cluster")

    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots()

    sns.boxplot(
        data=filtered_df,
        x="cluster",
        y="avg_session_duration_min",
        palette="Set1",
        ax=ax
    )

    ax.set_title("Session Duration by Cluster")

    st.pyplot(fig)


# Chart Row 3


st.subheader("📈 Engagement Trends")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots()

    sns.histplot(
        filtered_df["daily_active_minutes"],
        bins=40,
        kde=True,
        ax=ax
    )

    ax.set_title("Daily Active Minutes Distribution")

    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=filtered_df,
        x="sessions_per_week",
        y="engagement_score",
        hue="cluster",
        palette="Set2",
        ax=ax
    )

    ax.set_title("Sessions vs Engagement")

    st.pyplot(fig)


# PCA Visualization


st.subheader("🧠 PCA Cluster Visualization")

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=filtered_df,
    x="pca1",
    y="pca2",
    hue="cluster",
    palette="Set2",
    ax=ax
)

ax.set_title("User Segmentation Using PCA")

st.pyplot(fig)


# Cluster Summary Table


st.subheader("📋 Cluster Behavior Summary")

cluster_summary = filtered_df.groupby("cluster")[features].mean()

st.dataframe(cluster_summary)


# Download Data


st.subheader("📥 Download Segmented Data")

csv = filtered_df.to_csv(index=False)

st.download_button(
    "Download Data",
    csv,
    "filtered_user_segmentation.csv",
    "text/csv"
)