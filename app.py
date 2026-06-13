import streamlit as st
import pandas as pd
import sqlite3
import os

# ==========================================
# 1. GLOBAL STREAMLIT APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title=" PragyanAI AI Agentic AI - Karnataka College Matrix Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DATABASE INITIALIZATION LAYER (SQL)
# ==========================================
def init_db():
    """Initializes the relational database structure if it does not exist."""
    conn = sqlite3.connect("matrix_records.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS colleges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            college_name TEXT,
            city TEXT,
            district TEXT,
            address TEXT,
            dept TEXT,
            intake INTEGER,
            intake_year INTEGER,
            website TEXT DEFAULT 'N/A',
            naac TEXT DEFAULT 'N/A',
            nba TEXT DEFAULT 'N/A',
            nirf_rank TEXT DEFAULT 'N/A'
        )
    """)
    conn.commit()
    conn.close()

# Trigger relational schema synchronization
init_db()

# ==========================================
# 3. GLOBAL STATE CACHE INITIALIZATION
# ==========================================
# Keeps parsed and enriched states consistent across page files
if "shared_dataframe" not in st.session_state:
    # Attempt to reload existing entries from local DB if they exist
    try:
        conn = sqlite3.connect("matrix_records.db")
        existing_df = pd.read_sql_query("SELECT * FROM colleges", conn)
        conn.close()
        if not existing_df.empty:
            st.session_state["shared_dataframe"] = existing_df
        else:
            st.session_state["shared_dataframe"] = pd.DataFrame()
    except Exception:
        st.session_state["shared_dataframe"] = pd.DataFrame()

# ==========================================
# 4. MAIN USER INTERFACE VIEW
# ==========================================
st.title("🎓 Karnataka Professional Engineering Matrix Hub")
st.subheader("Unified Automation, Extraction, and Analytics Workspace")

st.markdown("""
---
This production-ready portal bridges the gap between layout-heavy document extraction pipelines and actionable institutional analytics.

### 🛠️ Core Functional Matrix Panels:
* **1. Data Extraction Engine (`pages/1_Data_Extraction.py`)**: Upload original seat allocation PDFs, parse dense structural tables natively with layout-aware tools matching your Google Colab workspace, specify the **Intake Year**, and commit structural tuples straight into a relational **SQLite Database**.
* **2. Profile Enrichment Matrix (`pages/2_Profile_Enrichment.py`)**: Enhance structural baseline records dynamically using programmatic web search fallback crawlers to append missing institutional meta-tags: **Website URLs, NAAC ratings, NBA statuses, and NIRF rankings**.
* **3. Analytics Dashboard (`pages/3_Analytics_Dashboard.py`)**: Run interactive Business Intelligence queries. Slice records by geographical district parameters, isolate specific branch distributions (such as **CSE**), and run volumetric analysis over available intake seats.
* **4. RAG Chat Explorer (`pages/4_RAG_Chat_Explorer.py`)**: Talk directly to your ingested data matrix using an enterprise Retrieval-Augmented Generation ecosystem powered by multiple LLM routers (**Qwen-2.5-32b, Llama-3.1-8b, Llama-3.3-70b**) combined with external web scraper fallback networks.
""")

# ==========================================
# 5. CURRENT DATA METRIC STATUS CARD
# ==========================================
st.markdown("### 📊 Database Operational Status")
df_active = st.session_state["shared_dataframe"]

if not df_active.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Ingested Matrix Rows", value=f"{len(df_active):,}")
    with col2:
        st.metric(label="Unique Institutes Accounted", value=f"{df_active['college_name'].nunique():,}")
    with col3:
        st.metric(label="Total Accounted Intake Seats Capacity", value=f"{df_active['intake'].sum():,}")
        
    st.info("💡 Data is actively synchronized in the operational pipeline state context. Explore the panels using the sidebar menu above.")
else:
    st.warning("⚠️ The underlying database index is currently empty. Head to the **1_Data_Extraction** page inside the sidebar layout panel to upload your source seat allocation document matrix.")

# ==========================================
# 6. APP ENVIRONMENT CHECKS
# ==========================================
st.sidebar.markdown("### 🔌 System Infrastructure Config")
if "GROQ_API_KEY" in os.environ or os.getenv("GROQ_API_KEY"):
    st.sidebar.success("Groq API Token Detected Natively")
else:
    st.sidebar.warning("Groq API Key not found in system variables. You can enter it manually inside the pipeline worker pages.")
