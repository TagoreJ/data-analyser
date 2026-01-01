# =========================================================
# Data Analyst AI Pro v2.2 – PRODUCTION READY
# Windows | Python 3.13 | OpenRouter | Request-based AI
# =========================================================
import certifi
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import io
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

# =========================================================
# CONFIG
# =========================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(
    page_title="Data Analyst AI Pro",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================

for key in ["df", "messages", "auto_insights"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "df" else []

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def call_ai(prompt, api_key, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    r = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    else:
        return f"AI Error: {r.text[:200]}"

def auto_statistical_insights(df):
    insights = []

    insights.append(f"Dataset has {df.shape[0]:,} rows and {df.shape[1]} columns.")

    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"There are {missing} missing values that may need treatment.")

    dup = df.duplicated().sum()
    if dup > 0:
        insights.append(f"{dup} duplicate records detected.")

    num_cols = df.select_dtypes(include="number")
    if not num_cols.empty:
        skewed = num_cols.skew().abs().sort_values(ascending=False)
        if skewed.iloc[0] > 1:
            insights.append(f"Column '{skewed.index[0]}' is highly skewed.")

        corr = num_cols.corr()
        strong = [
            (c1, c2, corr.loc[c1, c2])
            for c1 in corr.columns
            for c2 in corr.columns
            if c1 != c2 and abs(corr.loc[c1, c2]) > 0.7
        ]
        if strong:
            c1, c2, v = strong[0]
            insights.append(f"Strong correlation found between {c1} and {c2} ({v:.2f}).")

    return insights

def ai_auto_insights(df, api_key, model):
    preview = df.head(30).to_csv(index=False)

    prompt = f"""
You are a senior data analyst.
Analyze the dataset below and produce 5 executive-level insights.

Dataset preview:
{preview}

Focus on:
- Trends
- Risks
- Business implications
- Data quality
"""

    return call_ai(prompt, api_key, model)

def plot_to_bytes(fig):
    try:
        return fig.to_image(format="png", width=900, height=500)
    except:
        return None

def create_pdf(df, title, insights, charts):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, ParagraphStyle(
        "title", fontSize=26, alignment=TA_CENTER, textColor=colors.HexColor("#1f77b4")
    )))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %b %Y')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Key Insights", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(f"• {i}", styles["Normal"]))

    story.append(PageBreak())

    table_data = [df.columns.tolist()] + df.head(10).values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f77b4")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
    ]))
    story.append(table)

    story.append(PageBreak())

    for img in charts:
        if img:
            story.append(Image(io.BytesIO(img), width=6*inch, height=4*inch))
            story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input("OpenRouter API Key", type="password")
    model = st.selectbox("Model", [
        "amazon/nova-2-lite-v1:free",
        "arcee-ai/trinity-mini:free",
        "z-ai/glm-4.5-air:free"
    ])

    if st.button("Reset App"):
        st.session_state.clear()
        st.rerun()

# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Upload",
    "📊 Dashboard",
    "🧠 Auto Insights",
    "💬 AI Chat",
    "📄 Report"
])

# =========================================================
# TAB 1 – UPLOAD
# =========================================================

with tab1:
    file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.session_state.df = df
        st.success("Data loaded successfully")

        st.dataframe(df.head(), use_container_width=True)

# =========================================================
# TAB 2 – DASHBOARD
# =========================================================

with tab2:
    if st.session_state.df is None:
        st.info("Upload data first")
    else:
        df = st.session_state.df
        num = df.select_dtypes(include="number").columns

        if len(num) >= 1:
            fig = px.histogram(df, x=num[0], title=f"{num[0]} Distribution")
            st.plotly_chart(fig, use_container_width=True)

        if len(num) >= 2:
            fig = px.scatter(df, x=num[0], y=num[1], title="Scatter Analysis")
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 – AUTO INSIGHTS
# =========================================================

with tab3:
    if st.session_state.df is None:
        st.info("Upload data first")
    else:
        df = st.session_state.df

        if st.button("🔍 Generate Auto Insights"):
            st.session_state.auto_insights = auto_statistical_insights(df)

            if api_key:
                ai_insight = ai_auto_insights(df, api_key, model)
                st.session_state.auto_insights.append(ai_insight)

        for ins in st.session_state.auto_insights:
            st.markdown(f"✅ {ins}")

# =========================================================
# TAB 4 – AI CHAT
# =========================================================

with tab4:
    if st.session_state.df is None:
        st.info("Upload data first")
    elif not api_key:
        st.warning("Enter API key")
    else:
        df = st.session_state.df

        for m in st.session_state.messages[-10:]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if q := st.chat_input("Ask about your data"):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)

            context = df.head(40).to_csv(index=False)
            prompt = f"""
Analyze the dataset below and answer clearly.

{context}

Question: {q}
"""

            answer = call_ai(prompt, api_key, model)

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

# =========================================================
# TAB 5 – REPORT
# =========================================================

with tab5:
    if st.session_state.df is None:
        st.info("Upload data first")
    else:
        title = st.text_input("Report Title", "Data Analysis Report")

        if st.button("Generate PDF"):
            df = st.session_state.df
            insights = auto_statistical_insights(df)

            figs = []
            num = df.select_dtypes(include="number").columns
            if len(num) >= 1:
                figs.append(plot_to_bytes(px.histogram(df, x=num[0])))

            pdf = create_pdf(df, title, insights, figs)

            st.download_button(
                "Download PDF",
                pdf,
                file_name=f"{title}.pdf",
                mime="application/pdf"
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption("📊 Data Analyst AI Pro v2.2 | Stable | Production Ready | OpenRouter")
