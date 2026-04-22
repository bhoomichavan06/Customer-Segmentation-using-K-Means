
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}
div[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.06);
}
.graph-card {
    background: white;
    padding: 15px;
    border-radius: 14px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.06);
}
h1,h2,h3 {
    color: #222;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Wholesale customers data.csv")
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("⚙️ Controls")
k = st.sidebar.slider("Select Clusters", 2, 10, 4)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("📊 Customer Segmentation Dashboard")
st.caption("Wholesale Customers Data Analysis")

# ---------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------
features = df.drop(["Channel", "Region"], axis=1)

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled)

df["Cluster"] = clusters

score = silhouette_score(scaled, clusters)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled)

pca_df = pd.DataFrame({
    "PCA1": pca_data[:,0],
    "PCA2": pca_data[:,1],
    "Cluster": clusters.astype(str)
})

# ---------------------------------------------------
# TOP METRICS
# ---------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Customers", len(df))

with c2:
    st.metric("Features", len(features.columns))

with c3:
    st.metric("Clusters", k)

with c4:
    st.metric("Silhouette Score", round(score,3))

st.markdown("")

# ---------------------------------------------------
# ROW 1
# ---------------------------------------------------
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### Cluster Visualization")

    fig1 = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        template="simple_white",
        height=380
    )
    fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### Cluster Distribution")

    fig2 = px.histogram(
        df,
        x="Cluster",
        color="Cluster",
        template="simple_white",
        height=380
    )
    fig2.update_layout(showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# ROW 2
# ---------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown("### Feature Correlation")

    corr = features.corr()

    fig3 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        height=450
    )
    fig3.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown("### Average Spending by Cluster")

    summary = df.groupby("Cluster")[features.columns].mean().T

    fig4 = go.Figure()

    for col in summary.columns:
        fig4.add_trace(go.Bar(
            x=summary.index,
            y=summary[col],
            name=f"Cluster {col}"
        ))

    fig4.update_layout(
        barmode="group",
        template="simple_white",
        height=450,
        margin=dict(l=10,r=10,t=10,b=10)
    )

    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

