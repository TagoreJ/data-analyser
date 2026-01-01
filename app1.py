# Ultimate Data Analysis Platform v2.1 (Professional Grade - FIXED)
# Expert Data Analyst AI Assistant with Streamlit
# Windows + Python 3.13 Compatible | Production Ready

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from openai import OpenAI
from datetime import datetime
import io
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

import warnings
warnings.filterwarnings("ignore")

# ================================================
# CUSTOM HTML/CSS STYLING
# ================================================

CUSTOM_HTML_CSS = """
<style>
    :root {
        --primary: #1f77b4;
        --secondary: #ff7f0e;
        --success: #2ca02c;
        --danger: #d62728;
        --info: #17a2b8;
        --light: #f8f9fa;
        --dark: #212529;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    
    .insight-box {
        background: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #1a1a1a; 
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #664d03
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .code-block {
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
</style>
"""

# ================================================
# PAGE CONFIG & THEME
# ================================================

st.set_page_config(
    page_title="Data Analyst AI Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CUSTOM_HTML_CSS, unsafe_allow_html=True)

# ================================================
# HEADER WITH GRADIENT
# ================================================

st.markdown("""
<div class="header-gradient">
    <h1>📊 Data Analyst AI Pro v2.1</h1>
    <p style="font-size: 16px; margin: 10px 0;">Expert-Level Data Analysis • AI Chat • Advanced Visualizations • PDF Reports</p>
    <p style="font-size: 12px; opacity: 0.8;">Powered by OpenRouter Free AI • No Cost • Full Power</p>
</div>
""", unsafe_allow_html=True)

# ================================================
# SIDEBAR CONFIGURATION
# ================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key
    api_key = st.text_input(
        "🔑 OpenRouter API Key",
        type="password",
        help="Get free key at https://openrouter.ai/keys"
    )
    
    # Model Selection
    st.markdown("### 🤖 AI Model")
    models = {
        "⚡ Nova 2 Lite (Fastest)": "amazon/nova-2-lite-v1:free",
        "🎯 Trinity Mini (Best)": "arcee-ai/trinity-mini:free",
        "👨‍💻 Kat Coder Pro": "kwaipilot/kat-coder-pro:free",
        "🌍 GLM-4.5 Air": "z-ai/glm-4.5-air:free",
    }
    model_id = models[st.selectbox("Select Model", list(models.keys()))]
    
    # Analysis Settings
    st.markdown("### 🔧 Analysis Settings")
    confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
    correlation_threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5, 0.1)
    
    # Session Controls
    st.markdown("### 🗑️ Session")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset"):
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("📥 Export Session"):
            st.info("Session export feature coming soon")
    
    st.markdown("---")
    st.caption("Made with ❤️ for Data Professionals")

# ================================================
# SESSION STATE INITIALIZATION
# ================================================

if "df" not in st.session_state:
    st.session_state.df = None
if "df_stats" not in st.session_state:
    st.session_state.df_stats = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "insights" not in st.session_state:
    st.session_state.insights = []
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ================================================
# UTILITY FUNCTIONS
# ================================================

def calculate_advanced_statistics(df):
    """Calculate comprehensive statistics"""
    stats_dict = {
        "shape": df.shape,
        "memory": df.memory_usage(deep=True).sum() / 1024**2,
        "numeric_cols": df.select_dtypes(include='number').columns.tolist(),
        "categorical_cols": df.select_dtypes(include='object').columns.tolist(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_stats": df.describe().to_dict(),
    }
    return stats_dict

def detect_anomalies(df, numeric_cols):
    """Detect anomalies using IQR method"""
    anomalies = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col].tolist()
        anomalies[col] = {
            "count": len(outliers),
            "percentage": len(outliers) / len(df) * 100,
            "bounds": (lower, upper)
        }
    return anomalies

def extract_code_from_answer(answer_text):
    """Safely extract Python code from LLM response"""
    try:
        tick = chr(96)
        fence = tick + tick + tick
        code_block = answer_text
        if fence in answer_text:
            parts = answer_text.split(fence)
            if len(parts) >= 2:
                code_block = parts[1]
        code_lines = code_block.splitlines()
        if code_lines and code_lines[0].strip().lower().startswith("python"):
            code_lines = code_lines[1:]
        code = "\n".join(code_lines).strip()
        return code
    except Exception:
        return ""

def plotly_to_bytes(fig):
    """Convert plotly figure to PNG bytes"""
    try:
        return fig.to_image(format="png", width=1000, height=600)
    except Exception:
        return None

def create_professional_pdf(df, title, insights, charts_bytes):
    """Create professional PDF report with ReportLab"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=10,
        alignment=TA_CENTER
    )
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Insights
    if insights:
        story.append(Paragraph("AI Insights & Analysis", styles['Heading2']))
        for insight in insights[:5]:
            try:
                story.append(Paragraph(f"• {str(insight)[:200]}", styles['Normal']))
            except:
                pass
        story.append(Spacer(1, 20))
    
    # Data Preview Table
    if df is not None and not df.empty:
        story.append(Paragraph("Data Overview (First 10 Rows)", styles['Heading2']))
        try:
            data = [df.columns.tolist()[:5]] + df.head(10).values.tolist()[:5]
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            story.append(table)
            story.append(PageBreak())
        except:
            pass
    
    # Charts
    for i, img_bytes in enumerate(charts_bytes):
        if img_bytes:
            try:
                story.append(Paragraph(f"Visualization {i+1}", styles['Heading2']))
                img = Image(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                if i % 2 == 1:
                    story.append(PageBreak())
            except:
                pass
    
    doc.build(story)
    return buffer.getvalue()

# ================================================
# TABS CREATION
# ================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📥 Upload & Explore",
    "📊 Dashboard",
    "🔍 Deep Analysis",
    "💬 AI Chat",
    "✏️ Edit Data",
    "🧠 Advanced Analytics",
    "📄 Reports"
])

# ================================================
# TAB 1: UPLOAD & EXPLORE
# ================================================

with tab1:
    st.header("📥 Data Upload & Exploration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader(
            "📁 Upload CSV, Excel, or JSON file",
            type=["csv", "xlsx", "xls", "json"],
            help="Maximum 200MB"
        )
    
    with col2:
        sample_data = st.checkbox("📋 Use Sample Data")
    
    if sample_data:
        df = pd.DataFrame({
            'ID': range(1, 101),
            'Sales': np.random.randint(1000, 10000, 100),
            'Profit': np.random.randint(100, 2000, 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Category': np.random.choice(['Electronics', 'Clothing', 'Food'], 100),
            'Date': pd.date_range('2024-01-01', periods=100, freq='D')
        })
        st.session_state.df = df
        st.session_state.df_stats = calculate_advanced_statistics(df)
        st.success("✅ Sample data loaded!")
    elif uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            elif uploaded.name.endswith('.json'):
                df = pd.read_json(uploaded)
            else:
                df = pd.read_excel(uploaded)
            
            st.session_state.df = df
            st.session_state.df_stats = calculate_advanced_statistics(df)
            st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Display data statistics
    if st.session_state.df is not None:
        df = st.session_state.df
        stats = st.session_state.df_stats
        
        st.markdown("### 📈 Dataset Statistics")
        
        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("📊 Rows", f"{stats['shape'][0]:,}")
        with m2:
            st.metric("📋 Columns", stats['shape'][1])
        with m3:
            st.metric("💾 Size (MB)", f"{stats['memory']:.2f}")
        with m4:
            missing_count = sum(1 for v in stats['missing_pct'].values() if v > 0)
            st.metric("❌ Missing Cols", missing_count)
        with m5:
            st.metric("🔄 Duplicates", stats['duplicates'])
        
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Data Quality Report
        st.markdown("### 🔍 Data Quality Report")
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_str = ", ".join(stats['numeric_cols'][:5]) if stats['numeric_cols'] else "None"
            st.markdown(f"""
            <div class="insight-box">
                <strong>✅ Numeric Columns:</strong><br>
                {numeric_str}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            categorical_str = ", ".join(stats['categorical_cols'][:5]) if stats['categorical_cols'] else "None"
            st.markdown(f"""
            <div class="insight-box">
                <strong>🏷️ Categorical Columns:</strong><br>
                {categorical_str}
            </div>
            """, unsafe_allow_html=True)

# ================================================
# TAB 2: DASHBOARD
# ================================================

with tab2:
    st.header("📊 Interactive Dashboard")
    
    if st.session_state.df is None:
        st.info("👈 Upload data in the 'Upload & Explore' tab first")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        
        if num_cols:
            # Dashboard Controls
            st.markdown("### 🎛️ Dashboard Controls")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Distribution", "Scatter", "Box Plot", "Heatmap", "Time Series"]
                )
            with col2:
                x_col = st.selectbox("X Axis", num_cols)
            with col3:
                if len(num_cols) > 1:
                    y_col = st.selectbox("Y Axis", num_cols[1:])
                else:
                    y_col = num_cols[0]
            
            # Generate charts
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                if chart_type == "Distribution":
                    fig = px.histogram(
                        df,
                        x=x_col,
                        nbins=30,
                        title=f"{x_col} Distribution",
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Box Plot":
                    fig = px.box(
                        df,
                        y=x_col,
                        title=f"{x_col} Box Plot",
                        color_discrete_sequence=['#764ba2']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_viz2:
                if chart_type == "Scatter" and len(num_cols) > 1:
                    if cat_cols:
                        color_col = st.selectbox("Color by", cat_cols, key="scatter_color")
                        fig = px.scatter(
                            df,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            hover_data=df.columns[:5],
                            title=f"{x_col} vs {y_col}",
                        )
                    else:
                        fig = px.scatter(
                            df,
                            x=x_col,
                            y=y_col,
                            title=f"{x_col} vs {y_col}",
                        )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Heatmap" and len(num_cols) > 1:
                    fig = px.imshow(
                        df[num_cols].corr(),
                        text_auto=True,
                        title="Correlation Heatmap",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Summary Statistics
            st.markdown("### 📊 Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for dashboard visualization")

# ================================================
# TAB 3: DEEP ANALYSIS
# ================================================

with tab3:
    st.header("🔍 Deep Statistical Analysis")
    
    if st.session_state.df is None:
        st.info("👈 Upload data first")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include='number').columns.tolist()
        
        if num_cols:
            # Anomaly Detection
            st.markdown("### 🚨 Anomaly Detection (IQR Method)")
            anomalies = detect_anomalies(df, num_cols)
            
            col1, col2 = st.columns(2)
            for idx, (col, anom) in enumerate(anomalies.items()):
                if idx % 2 == 0:
                    with col1:
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>{col}</strong><br>
                            Outliers: {anom['count']} ({anom['percentage']:.2f}%)<br>
                            Bounds: [{anom['bounds'][0]:.2f}, {anom['bounds'][1]:.2f}]
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with col2:
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>{col}</strong><br>
                            Outliers: {anom['count']} ({anom['percentage']:.2f}%)<br>
                            Bounds: [{anom['bounds'][0]:.2f}, {anom['bounds'][1]:.2f}]
                        </div>
                        """, unsafe_allow_html=True)
            
            # Correlation Analysis
            st.markdown("### 🔗 Correlation Analysis")
            if len(num_cols) > 1:
                corr_matrix = df[num_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    title="Pearson Correlation Matrix",
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Find strong correlations
                st.markdown("**Strong Correlations (>0.7):**")
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corr.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j]
                            ))
                if strong_corr:
                    for col1, col2, corr in strong_corr:
                        st.markdown(f"- **{col1}** ↔ **{col2}**: {corr:.3f}")
                else:
                    st.info("No strong correlations found")
        else:
            st.info("No numeric columns available for analysis")

# ================================================
# TAB 4: AI CHAT
# ================================================

with tab4:
    st.header("💬 Chat with Your Data")
    
    if st.session_state.df is None:
        st.info("👈 Upload data first")
    elif not api_key:
        st.warning("⚠️ Enter OpenRouter API key in sidebar")
    else:
        df = st.session_state.df
        
        # Chat history
        for msg in st.session_state.messages[-10:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything about your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                try:
                    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                    
                    ctx = df.head(50).to_csv(index=False)
                    full_prompt = f"""You are an expert data analyst. Analyze this data and answer the question.
Data (CSV):
{ctx}

Data Stats:
- Rows: {len(df)}, Columns: {len(df.columns)}
- Numeric: {df.select_dtypes(include='number').columns.tolist()}
- Categorical: {df.select_dtypes(include='object').columns.tolist()}

Question: {prompt}

Provide:
1. Direct answer
2. Key insights
3. Relevant code (if needed, wrap in triple backticks python)"""
                    
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": full_prompt}],
                        stream=True,
                    )
                    
                    answer = ""
                    ph = st.empty()
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            answer += chunk.choices[0].delta.content
                            ph.markdown(answer + "▌")
                    
                    ph.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)[:100]}")

# ================================================
# TAB 5: EDIT DATA
# ================================================

with tab5:
    st.header("✏️ Data Editor")
    
    if st.session_state.df is None:
        st.info("👈 Upload data first")
    else:
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes"):
                st.session_state.df = edited_df
                st.success("✅ Changes saved!")
        with col2:
            if st.button("🔄 Revert"):
                st.rerun()

# ================================================
# TAB 6: ADVANCED ANALYTICS
# ================================================

with tab6:
    st.header("🧠 Advanced Analytics")
    
    if st.session_state.df is None:
        st.info("👈 Upload data first")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include='number').columns.tolist()
        
        if len(num_cols) > 2:
            # PCA Analysis
            if st.checkbox("🔵 PCA (Dimensionality Reduction)"):
                try:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df[num_cols])
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(scaled_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pca_data[:, 0],
                        y=pca_data[:, 1],
                        mode='markers',
                        marker=dict(size=8, color=pca_data[:, 0], colorscale='Viridis'),
                    ))
                    fig.update_layout(
                        title=f"PCA (Explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance)",
                        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"PCA Error: {str(e)[:50]}")
            
            # K-Means Clustering
            if st.checkbox("🎯 K-Means Clustering"):
                try:
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df[num_cols])
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    fig = px.scatter(
                        x=df[num_cols[0]],
                        y=df[num_cols[1]],
                        color=clusters,
                        title=f"K-Means Clustering (K={n_clusters})",
                        labels={'x': num_cols[0], 'y': num_cols[1], 'color': 'Cluster'},
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Clustering Error: {str(e)[:50]}")
        else:
            st.info("Need at least 3 numeric columns for advanced analytics")

# ================================================
# TAB 7: REPORTS
# ================================================

with tab7:
    st.header("📄 Report Generation")
    
    if st.session_state.df is None:
        st.info("👈 Upload data first")
    else:
        df = st.session_state.df
        
        st.markdown("### 📋 Report Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            report_title = st.text_input("Report Title", "Data Analysis Report")
        with col2:
            report_format = st.selectbox("Format", ["PDF"])
        
        if st.button("🚀 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                charts = []
                num_cols = df.select_dtypes(include='number').columns.tolist()
                
                try:
                    if len(num_cols) >= 1:
                        fig = px.histogram(df, x=num_cols[0], title=f"{num_cols[0]} Distribution")
                        chart_bytes = plotly_to_bytes(fig)
                        if chart_bytes:
                            charts.append(chart_bytes)
                    if len(num_cols) >= 2:
                        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title="Scatter Plot")
                        chart_bytes = plotly_to_bytes(fig)
                        if chart_bytes:
                            charts.append(chart_bytes)
                except Exception as e:
                    st.warning(f"Chart generation issue: {str(e)[:50]}")
                
                insights = [
                    f"Dataset contains {df.shape[0]:,} records and {df.shape[1]} variables",
                    f"Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.2f}%)",
                    f"Duplicate records: {df.duplicated().sum()}",
                ]
                
                try:
                    pdf = create_professional_pdf(df, report_title, insights, charts)
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf,
                        file_name=f"{report_title.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Report generated successfully!")
                except Exception as e:
                    st.error(f"PDF generation error: {str(e)[:100]}")

# ================================================
# FOOTER
# ================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7;">
    <p>📊 Data Analyst AI Pro v2.1 | Built with Streamlit | Powered by OpenRouter</p>
    <p style="font-size: 12px;">✨ Professional Grade • Production Ready • Zero Cost</p>
</div>
""", unsafe_allow_html=True)

